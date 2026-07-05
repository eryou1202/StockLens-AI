from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from src.data.symbol_mapper import SymbolMapper
from src.portfolio.position_schema import Position, PositionStatus
from src.storage.sqlite_store import SQLiteSignalStore


class PositionManager:
    def __init__(self, database_path: str):
        self.database_path = Path(database_path)
        SQLiteSignalStore(str(self.database_path))

    def add_position(self, position: Position) -> int:
        symbol = SymbolMapper.normalize(position.symbol)
        if position.status == PositionStatus.OPEN and self.get_open_position(symbol) is not None:
            raise ValueError(f"{symbol} 已存在 open 持仓，请勿重复添加。")

        now = datetime.now().isoformat()
        values = (
            symbol,
            position.stock_name,
            position.entry_date.isoformat(),
            position.entry_price,
            position.position_size,
            position.entry_reason,
            position.entry_signal_id,
            position.entry_action,
            position.stop_loss_price,
            position.take_profit_price,
            position.max_holding_days,
            position.status.value,
            self._iso_or_none(position.exit_date),
            position.exit_price,
            position.exit_reason,
            now,
            now,
            json.dumps(position.metadata, ensure_ascii=False, default=str),
        )
        try:
            with sqlite3.connect(self.database_path) as connection:
                cursor = connection.execute(
                    """
                    INSERT INTO positions (
                        symbol, stock_name, entry_date, entry_price, position_size,
                        entry_reason, entry_signal_id, entry_action,
                        stop_loss_price, take_profit_price, max_holding_days, status,
                        exit_date, exit_price, exit_reason, created_at, updated_at, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    values,
                )
                return int(cursor.lastrowid)
        except sqlite3.IntegrityError as exc:
            raise ValueError(f"{symbol} 已存在 open 持仓，请勿重复添加。") from exc

    def list_positions(self, status: str | None = "open") -> list[Position]:
        query = "SELECT * FROM positions"
        params: tuple[Any, ...] = ()
        if status is not None and status != "all":
            normalized_status = PositionStatus(status).value
            query += " WHERE status = ?"
            params = (normalized_status,)
        query += " ORDER BY id DESC"
        with sqlite3.connect(self.database_path) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(query, params).fetchall()
        return [self._row_to_position(row) for row in rows]

    def get_open_position(self, symbol: str) -> Position | None:
        normalized_symbol = SymbolMapper.normalize(symbol)
        with sqlite3.connect(self.database_path) as connection:
            connection.row_factory = sqlite3.Row
            row = connection.execute(
                "SELECT * FROM positions WHERE symbol = ? AND status = 'open' "
                "ORDER BY id DESC LIMIT 1",
                (normalized_symbol,),
            ).fetchone()
        return self._row_to_position(row) if row is not None else None

    def close_position(
        self,
        symbol: str,
        exit_price: float,
        exit_date: datetime,
        exit_reason: str | None = None,
    ) -> None:
        if exit_price <= 0:
            raise ValueError("exit_price 必须大于 0。")
        normalized_symbol = SymbolMapper.normalize(symbol)
        now = datetime.now().isoformat()
        with sqlite3.connect(self.database_path) as connection:
            row = connection.execute(
                "SELECT id FROM positions WHERE symbol = ? AND status = 'open' "
                "ORDER BY id DESC LIMIT 1",
                (normalized_symbol,),
            ).fetchone()
            if row is None:
                raise ValueError(f"未找到 {normalized_symbol} 的 open 持仓。")
            connection.execute(
                """
                UPDATE positions
                SET status = 'closed', exit_date = ?, exit_price = ?,
                    exit_reason = ?, updated_at = ?
                WHERE id = ?
                """,
                (exit_date.isoformat(), exit_price, exit_reason, now, row[0]),
            )

    @classmethod
    def _row_to_position(cls, row: sqlite3.Row) -> Position:
        metadata: dict[str, Any] = {}
        try:
            parsed = json.loads(row["metadata_json"] or "{}")
            if isinstance(parsed, dict):
                metadata = parsed
        except (TypeError, json.JSONDecodeError):
            metadata = {}
        return Position(
            id=row["id"],
            symbol=row["symbol"],
            stock_name=row["stock_name"],
            entry_date=cls._parse_datetime(row["entry_date"]),
            entry_price=row["entry_price"],
            position_size=row["position_size"],
            entry_reason=row["entry_reason"],
            entry_signal_id=row["entry_signal_id"],
            entry_action=row["entry_action"],
            stop_loss_price=row["stop_loss_price"],
            take_profit_price=row["take_profit_price"],
            max_holding_days=row["max_holding_days"],
            status=PositionStatus(row["status"]),
            exit_date=cls._parse_datetime(row["exit_date"]) if row["exit_date"] else None,
            exit_price=row["exit_price"],
            exit_reason=row["exit_reason"],
            metadata=metadata,
        )

    @staticmethod
    def _parse_datetime(value: str) -> datetime:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))

    @staticmethod
    def _iso_or_none(value: datetime | None) -> str | None:
        return value.isoformat() if value is not None else None
