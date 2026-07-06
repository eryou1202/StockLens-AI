from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from src.data.symbol_mapper import SymbolMapper
from src.data.symbol_name_resolver import SymbolNameResolver
from src.portfolio.position_schema import Position, PositionStatus
from src.storage.sqlite_store import SQLiteSignalStore


class PositionManager:
    WATCH_PRICE_THRESHOLD = 0.01

    def __init__(self, database_path: str):
        self.database_path = Path(database_path)
        SQLiteSignalStore(str(self.database_path))

    def add_position(self, position: Position) -> int:
        symbol = SymbolMapper.normalize(position.symbol)
        normalized = self._normalize_position(position, symbol)
        if not normalized.stock_name or normalized.stock_name.strip() in {"", "-", "未知名称"}:
            resolved_name = SymbolNameResolver(str(self.database_path)).resolve(symbol)
            if resolved_name:
                normalized = normalized.model_copy(update={"stock_name": resolved_name})
        if normalized.status == PositionStatus.OPEN and self.get_open_position(symbol):
            raise ValueError(f"{symbol} 已存在 open 持仓，请勿重复添加。")
        if normalized.status == PositionStatus.WATCH_ONLY and self.get_watch_position(symbol):
            raise ValueError(f"{symbol} 已存在 active watch_only 观察股。")

        now = datetime.now().isoformat()
        values = (
            symbol, normalized.stock_name, normalized.entry_date.isoformat(),
            normalized.entry_price, normalized.position_size, normalized.entry_reason,
            normalized.entry_signal_id, normalized.entry_action,
            normalized.stop_loss_price, normalized.take_profit_price,
            normalized.max_holding_days, normalized.status.value,
            self._iso_or_none(normalized.exit_date), normalized.exit_price,
            normalized.exit_reason, now, now,
            json.dumps(normalized.metadata, ensure_ascii=False, default=str),
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
                    """, values,
                )
                return int(cursor.lastrowid)
        except sqlite3.IntegrityError as exc:
            raise ValueError(f"{symbol} 已存在相同类型的活动持仓或观察股。") from exc

    def list_positions(self, status: str | None = "active") -> list[Position]:
        query = "SELECT * FROM positions"
        params: tuple[Any, ...] = ()
        if status in (None, "active"):
            query += " WHERE status IN ('open', 'watch_only')"
        elif status != "all":
            normalized_status = PositionStatus(status).value
            query += " WHERE status = ?"
            params = (normalized_status,)
        query += " ORDER BY id DESC"
        with sqlite3.connect(self.database_path) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(query, params).fetchall()
        return [self._row_to_position(row) for row in rows]

    def get_open_position(self, symbol: str) -> Position | None:
        return self._get_active(symbol, PositionStatus.OPEN)

    def get_watch_position(self, symbol: str) -> Position | None:
        return self._get_active(symbol, PositionStatus.WATCH_ONLY)

    def _get_active(self, symbol: str, status: PositionStatus) -> Position | None:
        normalized = SymbolMapper.normalize(symbol)
        with sqlite3.connect(self.database_path) as connection:
            connection.row_factory = sqlite3.Row
            row = connection.execute(
                "SELECT * FROM positions WHERE symbol = ? AND status = ? ORDER BY id DESC LIMIT 1",
                (normalized, status.value),
            ).fetchone()
        return self._row_to_position(row) if row else None

    def close_position(self, symbol: str, exit_price: float, exit_date: datetime,
                       exit_reason: str | None = None) -> None:
        if exit_price <= 0:
            raise ValueError("exit_price 必须大于 0。")
        normalized = SymbolMapper.normalize(symbol)
        now = datetime.now().isoformat()
        with sqlite3.connect(self.database_path) as connection:
            row = connection.execute(
                "SELECT id FROM positions WHERE symbol = ? AND status = 'open' ORDER BY id DESC LIMIT 1",
                (normalized,),
            ).fetchone()
            if not row:
                raise ValueError(f"未找到 {normalized} 的 open 持仓；watch_only 请使用删除观察股。")
            connection.execute(
                "UPDATE positions SET status='closed', exit_date=?, exit_price=?, "
                "exit_reason=?, updated_at=? WHERE id=?",
                (exit_date.isoformat(), exit_price, exit_reason, now, row[0]),
            )

    def remove_watch(self, symbol: str | None = None, position_id: int | None = None) -> int:
        if symbol is None and position_id is None:
            raise ValueError("symbol 和 position_id 至少提供一个。")
        clauses = ["status = 'watch_only'"]
        params: list[Any] = []
        if position_id is not None:
            clauses.append("id = ?")
            params.append(int(position_id))
        if symbol is not None:
            clauses.append("symbol = ?")
            params.append(SymbolMapper.normalize(symbol))
        with sqlite3.connect(self.database_path) as connection:
            cursor = connection.execute(f"DELETE FROM positions WHERE {' AND '.join(clauses)}", params)
            return int(cursor.rowcount)

    def cleanup_test_positions(self) -> int:
        now = datetime.now().isoformat()
        with sqlite3.connect(self.database_path) as connection:
            cursor = connection.execute(
                """
                UPDATE positions
                SET status='watch_only', entry_action=COALESCE(entry_action, 'cleanup_watch'),
                    updated_at=?
                WHERE status='open' AND entry_price <= ?
                """,
                (now, self.WATCH_PRICE_THRESHOLD),
            )
            return int(cursor.rowcount)

    def update_stock_name(self, position_id: int, stock_name: str) -> None:
        name = stock_name.strip()
        if not name:
            return
        with sqlite3.connect(self.database_path) as connection:
            connection.execute(
                "UPDATE positions SET stock_name=?, updated_at=? WHERE id=?",
                (name, datetime.now().isoformat(), position_id),
            )

    @classmethod
    def _normalize_position(cls, position: Position, symbol: str) -> Position:
        if position.entry_price <= cls.WATCH_PRICE_THRESHOLD:
            metadata = dict(position.metadata)
            metadata.setdefault("source_type", "manual_watch")
            metadata.setdefault("watch_only_reason", "entry_price_lte_0.01")
            return position.model_copy(update={
                "symbol": symbol,
                "status": PositionStatus.WATCH_ONLY,
                "entry_action": position.entry_action or "manual_watch",
                "position_size": None,
                "stop_loss_price": None,
                "take_profit_price": None,
                "metadata": metadata,
            })
        return position.model_copy(update={"symbol": symbol})

    @classmethod
    def _row_to_position(cls, row: sqlite3.Row) -> Position:
        try:
            metadata = json.loads(row["metadata_json"] or "{}")
            if not isinstance(metadata, dict):
                metadata = {}
        except (TypeError, json.JSONDecodeError):
            metadata = {}
        return Position(
            id=row["id"], symbol=row["symbol"], stock_name=row["stock_name"],
            entry_date=cls._parse_datetime(row["entry_date"]), entry_price=row["entry_price"],
            position_size=row["position_size"], entry_reason=row["entry_reason"],
            entry_signal_id=row["entry_signal_id"], entry_action=row["entry_action"],
            stop_loss_price=row["stop_loss_price"], take_profit_price=row["take_profit_price"],
            max_holding_days=row["max_holding_days"], status=PositionStatus(row["status"]),
            exit_date=cls._parse_datetime(row["exit_date"]) if row["exit_date"] else None,
            exit_price=row["exit_price"], exit_reason=row["exit_reason"], metadata=metadata,
        )

    @staticmethod
    def _parse_datetime(value: str) -> datetime:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))

    @staticmethod
    def _iso_or_none(value: datetime | None) -> str | None:
        return value.isoformat() if value else None
