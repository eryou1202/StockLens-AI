from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any


class ResearchModelRegistry:
    ALLOWED_STATUSES = {"research", "rejected", "archived"}

    def __init__(
        self,
        database_path: str = "data/models/registry/model_registry.sqlite",
        model_dir: str = "data/models/research",
    ) -> None:
        self.database_path = Path(database_path)
        self.model_dir = Path(model_dir)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def new_model_id(self) -> str:
        return f"research_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

    def model_path(self, model_id: str) -> Path:
        return self.model_dir / f"{model_id}.joblib"

    def register(self, record: dict[str, Any]) -> None:
        status = str(record.get("status", "research"))
        if status not in self.ALLOWED_STATUSES:
            raise ValueError(f"unsupported research registry status: {status}")
        fields = [
            "model_id", "model_name", "model_type", "target", "target_horizon",
            "dataset_path", "train_start", "train_end", "valid_start", "valid_end",
            "features_json", "metrics_json", "model_path", "status", "created_at", "notes",
        ]
        values = [record.get(field) for field in fields]
        with sqlite3.connect(self.database_path) as connection:
            connection.execute(
                f"INSERT INTO research_models ({', '.join(fields)}) VALUES "
                f"({', '.join('?' for _ in fields)})",
                values,
            )

    def list_models(self, limit: int = 100) -> list[dict[str, Any]]:
        with sqlite3.connect(self.database_path) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                "SELECT * FROM research_models ORDER BY created_at DESC LIMIT ?", (max(1, limit),)
            ).fetchall()
        return [self._decode(dict(row)) for row in rows]

    def get(self, model_id: str) -> dict[str, Any] | None:
        with sqlite3.connect(self.database_path) as connection:
            connection.row_factory = sqlite3.Row
            row = connection.execute(
                "SELECT * FROM research_models WHERE model_id=?", (model_id,)
            ).fetchone()
        return None if row is None else self._decode(dict(row))

    def _init_db(self) -> None:
        with sqlite3.connect(self.database_path) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS research_models (
                    model_id TEXT PRIMARY KEY,
                    model_name TEXT NOT NULL,
                    model_type TEXT NOT NULL,
                    target TEXT NOT NULL,
                    target_horizon INTEGER,
                    dataset_path TEXT NOT NULL,
                    train_start TEXT,
                    train_end TEXT NOT NULL,
                    valid_start TEXT NOT NULL,
                    valid_end TEXT NOT NULL,
                    features_json TEXT NOT NULL,
                    metrics_json TEXT NOT NULL,
                    model_path TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    notes TEXT
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS ix_research_models_status_created "
                "ON research_models(status, created_at)"
            )

    @staticmethod
    def dumps(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, allow_nan=False, default=str)

    @staticmethod
    def _decode(row: dict[str, Any]) -> dict[str, Any]:
        for column in ("features_json", "metrics_json"):
            try:
                row[column.removesuffix("_json")] = json.loads(row.get(column) or "null")
            except json.JSONDecodeError:
                row[column.removesuffix("_json")] = None
        return row
