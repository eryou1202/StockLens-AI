from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from src.feedback.feedback_schema import FutureLabel
from src.models.schemas import FinalDecision


class SQLiteSignalStore:
    """
    SQLite 存储层。

    初版保存完整 JSON 快照。
    """

    def __init__(self, db_path: str = "data/signals.sqlite"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS signal_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    as_of_time TEXT NOT NULL,
                    final_level TEXT NOT NULL,
                    final_score REAL NOT NULL,
                    final_decision_json TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS ix_signal_snapshots_symbol_as_of_time "
                "ON signal_snapshots(symbol, as_of_time)"
            )
            self._init_feedback_table(conn)

    @staticmethod
    def _init_feedback_table(conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS signal_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                signal_id TEXT,
                symbol TEXT NOT NULL,
                stock_name TEXT,
                as_of_time TEXT NOT NULL,
                entry_trade_date TEXT,
                entry_close REAL,
                future_trade_date_1d TEXT,
                future_trade_date_3d TEXT,
                future_trade_date_5d TEXT,
                future_trade_date_10d TEXT,
                future_close_1d REAL,
                future_close_3d REAL,
                future_close_5d REAL,
                future_close_10d REAL,
                future_return_1d REAL,
                future_return_3d REAL,
                future_return_5d REAL,
                future_return_10d REAL,
                future_max_drawdown_5d REAL,
                future_max_drawdown_10d REAL,
                hit_1d INTEGER,
                hit_3d INTEGER,
                hit_5d INTEGER,
                hit_10d INTEGER,
                feedback_status TEXT NOT NULL,
                error_message TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                metadata_json TEXT,
                UNIQUE(symbol, as_of_time)
            )
            """
        )

        # 兼容早期开发库：表存在但字段不完整时只增量补列。
        existing = {row[1] for row in conn.execute("PRAGMA table_info(signal_feedback)")}
        required_columns = {
            "signal_id": "TEXT",
            "stock_name": "TEXT",
            "entry_trade_date": "TEXT",
            "entry_close": "REAL",
            "future_trade_date_1d": "TEXT",
            "future_trade_date_3d": "TEXT",
            "future_trade_date_5d": "TEXT",
            "future_trade_date_10d": "TEXT",
            "future_close_1d": "REAL",
            "future_close_3d": "REAL",
            "future_close_5d": "REAL",
            "future_close_10d": "REAL",
            "future_return_1d": "REAL",
            "future_return_3d": "REAL",
            "future_return_5d": "REAL",
            "future_return_10d": "REAL",
            "future_max_drawdown_5d": "REAL",
            "future_max_drawdown_10d": "REAL",
            "hit_1d": "INTEGER",
            "hit_3d": "INTEGER",
            "hit_5d": "INTEGER",
            "hit_10d": "INTEGER",
            "feedback_status": "TEXT NOT NULL DEFAULT 'pending'",
            "error_message": "TEXT",
            "created_at": "TEXT NOT NULL DEFAULT ''",
            "updated_at": "TEXT NOT NULL DEFAULT ''",
            "metadata_json": "TEXT",
        }
        for column, definition in required_columns.items():
            if column not in existing:
                conn.execute(f"ALTER TABLE signal_feedback ADD COLUMN {column} {definition}")

        try:
            conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS ux_signal_feedback_symbol_as_of_time "
                "ON signal_feedback(symbol, as_of_time)"
            )
        except sqlite3.IntegrityError:
            # upsert_feedback 仍会手动查重；旧库中即使已有重复数据也不会阻断启动。
            pass

    def save_decision(self, decision: FinalDecision) -> None:
        as_of_time = decision.as_of_time.isoformat()
        values = (
            decision.final_level.value,
            decision.final_score,
            decision.model_dump_json(),
        )
        with self._connect() as conn:
            existing = conn.execute(
                "SELECT id FROM signal_snapshots "
                "WHERE symbol = ? AND as_of_time = ? ORDER BY id DESC LIMIT 1",
                (decision.symbol, as_of_time),
            ).fetchone()
            if existing:
                conn.execute(
                    """
                    UPDATE signal_snapshots
                    SET final_level = ?, final_score = ?, final_decision_json = ?
                    WHERE id = ?
                    """,
                    (*values, existing[0]),
                )
            else:
                conn.execute(
                    """
                    INSERT INTO signal_snapshots (
                        symbol, as_of_time, final_level, final_score, final_decision_json
                    )
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (decision.symbol, as_of_time, *values),
                )

    def list_recent_decisions(self, limit: int = 20) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT symbol, as_of_time, final_level, final_score, final_decision_json
                FROM signal_snapshots
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

        return [
            {
                "symbol": row[0],
                "as_of_time": row[1],
                "final_level": row[2],
                "final_score": row[3],
                "final_decision": json.loads(row[4]),
            }
            for row in rows
        ]

    def list_signals_for_feedback(self, limit: int | None = None) -> list[dict[str, Any]]:
        query = """
            SELECT s.id, s.symbol, s.as_of_time, s.final_level, s.final_score,
                   s.final_decision_json
            FROM signal_snapshots AS s
            LEFT JOIN signal_feedback AS f
              ON f.symbol = s.symbol AND f.as_of_time = s.as_of_time
            WHERE f.id IS NULL OR f.feedback_status IN ('pending', 'partial')
            ORDER BY s.id ASC
        """
        params: tuple[Any, ...] = ()
        if limit is not None:
            query += " LIMIT ?"
            params = (max(0, int(limit)),)

        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()

        signals: list[dict[str, Any]] = []
        for row in rows:
            try:
                payload = json.loads(row[5])
            except (TypeError, json.JSONDecodeError):
                payload = {}
            signals.append(
                {
                    "signal_id": str(row[0]),
                    "symbol": row[1],
                    "stock_name": payload.get("stock_name"),
                    "as_of_time": row[2],
                    "final_level": row[3],
                    "final_score": row[4],
                    "ai_view": payload.get("ai_view"),
                    "quant_view": payload.get("quant_view"),
                    "quant_decision": (payload.get("quant_result") or {}).get("quant_decision"),
                }
            )
        return signals

    def upsert_feedback(
        self,
        signal_id: str,
        label: FutureLabel,
        metadata: dict[str, Any] | None = None,
        as_of_time_key: str | None = None,
    ) -> None:
        now = datetime.now().isoformat()
        stored_as_of_time = as_of_time_key or label.as_of_time.isoformat()
        combined_metadata = dict(label.metadata)
        if metadata:
            combined_metadata.update(metadata)

        fields = [
            "signal_id", "symbol", "stock_name", "as_of_time",
            "entry_trade_date", "entry_close",
            "future_trade_date_1d", "future_trade_date_3d",
            "future_trade_date_5d", "future_trade_date_10d",
            "future_close_1d", "future_close_3d", "future_close_5d", "future_close_10d",
            "future_return_1d", "future_return_3d", "future_return_5d", "future_return_10d",
            "future_max_drawdown_5d", "future_max_drawdown_10d",
            "hit_1d", "hit_3d", "hit_5d", "hit_10d",
            "feedback_status", "error_message", "created_at", "updated_at", "metadata_json",
        ]
        values: list[Any] = [
            signal_id,
            label.symbol,
            label.stock_name,
            stored_as_of_time,
            self._iso_or_none(label.entry_trade_date),
            label.entry_close,
            self._iso_or_none(label.future_trade_date_1d),
            self._iso_or_none(label.future_trade_date_3d),
            self._iso_or_none(label.future_trade_date_5d),
            self._iso_or_none(label.future_trade_date_10d),
            label.future_close_1d,
            label.future_close_3d,
            label.future_close_5d,
            label.future_close_10d,
            label.future_return_1d,
            label.future_return_3d,
            label.future_return_5d,
            label.future_return_10d,
            label.future_max_drawdown_5d,
            label.future_max_drawdown_10d,
            self._bool_or_none(label.hit_1d),
            self._bool_or_none(label.hit_3d),
            self._bool_or_none(label.hit_5d),
            self._bool_or_none(label.hit_10d),
            label.feedback_status,
            label.error_message,
            now,
            now,
            json.dumps(combined_metadata, ensure_ascii=False, default=str),
        ]

        with self._connect() as conn:
            existing = conn.execute(
                "SELECT id, created_at FROM signal_feedback WHERE symbol = ? AND as_of_time = ? LIMIT 1",
                (label.symbol, stored_as_of_time),
            ).fetchone()
            if existing:
                update_fields = [field for field in fields if field != "created_at"]
                update_values = [value for field, value in zip(fields, values) if field != "created_at"]
                assignments = ", ".join(f"{field} = ?" for field in update_fields)
                conn.execute(
                    f"UPDATE signal_feedback SET {assignments} WHERE id = ?",
                    (*update_values, existing[0]),
                )
            else:
                placeholders = ", ".join("?" for _ in fields)
                conn.execute(
                    f"INSERT INTO signal_feedback ({', '.join(fields)}) VALUES ({placeholders})",
                    values,
                )

    def feedback_status_counts(self) -> dict[str, int]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT feedback_status, COUNT(*) FROM signal_feedback GROUP BY feedback_status"
            ).fetchall()
        return {str(status): int(count) for status, count in rows}

    @staticmethod
    def _iso_or_none(value: datetime | None) -> str | None:
        return value.isoformat() if value is not None else None

    @staticmethod
    def _bool_or_none(value: bool | None) -> int | None:
        return None if value is None else int(value)
