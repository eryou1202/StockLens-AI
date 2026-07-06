from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from src.data.market_data_provider import MarketDataProvider
from src.feedback.label_builder import FutureLabelBuilder
from src.recommendation.recommendation_schema import Recommendation
from src.storage.sqlite_store import SQLiteSignalStore
from src.tracking.tracking_schema import ManualVerdict, TrackedRecommendation, TrackingStatus


class RecommendationTracker:
    def __init__(self, database_path: str = "data/signals.sqlite",
                 market_data_provider: MarketDataProvider | None = None,
                 adjust_type: str = "qfq") -> None:
        self.database_path = Path(database_path)
        self.market_data_provider = market_data_provider
        self.adjust_type = adjust_type
        SQLiteSignalStore(str(self.database_path))

    def save_recommendations(self, recommendations: list[Recommendation]) -> int:
        now = datetime.now().isoformat()
        inserted = 0
        with sqlite3.connect(self.database_path) as conn:
            for item in recommendations:
                as_of = item.as_of_time.isoformat()
                existing = conn.execute(
                    "SELECT id FROM recommendation_tracking WHERE symbol=? AND as_of_time=? AND action=? LIMIT 1",
                    (item.symbol, as_of, item.action.value),
                ).fetchone()
                if existing:
                    continue
                conn.execute(
                    """
                    INSERT INTO recommendation_tracking (
                        symbol, stock_name, as_of_time, source_type, action, action_level,
                        confidence, final_score, ai_view, quant_decision, final_level,
                        current_price, suggested_horizon_days_json, reason_json, risks_json,
                        invalid_conditions_json, tracking_status, created_at, updated_at, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'tracking', ?, ?, ?)
                    """,
                    (
                        item.symbol, item.stock_name, as_of, item.source_type, item.action.value,
                        item.action_level.value, item.confidence, item.final_score, item.ai_view,
                        item.quant_decision, item.final_level, item.metadata.get("latest_close"),
                        self._json(item.suggested_horizon_days), self._json(item.reason),
                        self._json(item.risks), self._json(item.invalid_conditions), now, now,
                        self._json(item.metadata),
                    ),
                )
                inserted += 1
        return inserted

    def list_tracking(self, status: str | None = None) -> list[TrackedRecommendation]:
        query = "SELECT * FROM recommendation_tracking"
        params: tuple[Any, ...] = ()
        if status and status != "all":
            query += " WHERE tracking_status=?"
            params = (TrackingStatus(status).value,)
        query += " ORDER BY id DESC"
        with sqlite3.connect(self.database_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(query, params).fetchall()
        return [self._row(row) for row in rows]

    def update_future_performance(self) -> dict[str, int]:
        if self.market_data_provider is None:
            raise ValueError("更新未来表现需要 market_data_provider。")
        records = self.list_tracking(TrackingStatus.TRACKING.value)
        summary = {"found": len(records), "updated": 0, "tracking": 0, "complete": 0, "failed": 0}
        now = datetime.now().astimezone()
        builder = FutureLabelBuilder()
        for item in records:
            try:
                as_of = item.as_of_time
                comparable_now = now.replace(tzinfo=as_of.tzinfo) if as_of.tzinfo is not None else now.replace(tzinfo=None)
                if as_of > comparable_now:
                    summary["tracking"] += 1
                    continue
                end = min(as_of + timedelta(days=45), comparable_now)
                bundle = self.market_data_provider.get_bars(
                    item.symbol, as_of - timedelta(days=10), end, "1d", self.adjust_type
                )
                label = builder.build(item.symbol, item.stock_name, as_of, bundle)
                status = "failed" if label.feedback_status == "failed" else (
                    "complete" if label.feedback_status == "complete" else "tracking"
                )
                values = (
                    label.future_return_1d, label.future_return_3d, label.future_return_5d,
                    label.future_return_10d, label.future_max_drawdown_5d,
                    label.future_max_drawdown_10d, status, datetime.now().isoformat(), item.id,
                )
                with sqlite3.connect(self.database_path) as conn:
                    conn.execute(
                        """
                        UPDATE recommendation_tracking SET
                            future_return_1d=?, future_return_3d=?, future_return_5d=?,
                            future_return_10d=?, future_max_drawdown_5d=?,
                            future_max_drawdown_10d=?, tracking_status=?, updated_at=? WHERE id=?
                        """, values,
                    )
                summary["updated"] += 1
                summary[status] += 1
            except Exception as exc:
                metadata = dict(item.metadata)
                metadata["tracking_error"] = f"{type(exc).__name__}: {exc}"
                with sqlite3.connect(self.database_path) as conn:
                    conn.execute(
                        "UPDATE recommendation_tracking SET tracking_status='failed', metadata_json=?, updated_at=? WHERE id=?",
                        (self._json(metadata), datetime.now().isoformat(), item.id),
                    )
                summary["updated"] += 1
                summary["failed"] += 1
        return summary

    def mark_verdict(self, tracking_id: int, verdict: str, notes: str | None = None) -> None:
        value = ManualVerdict(verdict).value
        with sqlite3.connect(self.database_path) as conn:
            cursor = conn.execute(
                "UPDATE recommendation_tracking SET manual_verdict=?, manual_notes=?, updated_at=? WHERE id=?",
                (value, notes, datetime.now().isoformat(), int(tracking_id)),
            )
            if cursor.rowcount == 0:
                raise ValueError(f"未找到 tracking id={tracking_id}。")

    @classmethod
    def _row(cls, row: sqlite3.Row) -> TrackedRecommendation:
        return TrackedRecommendation(
            id=row["id"], symbol=row["symbol"], stock_name=row["stock_name"],
            as_of_time=cls._dt(row["as_of_time"]), source_type=row["source_type"],
            action=row["action"], action_level=row["action_level"], confidence=row["confidence"],
            final_score=row["final_score"], ai_view=row["ai_view"], quant_decision=row["quant_decision"],
            final_level=row["final_level"], current_price=row["current_price"],
            suggested_horizon_days=cls._list(row["suggested_horizon_days_json"]),
            reason=cls._list(row["reason_json"]), risks=cls._list(row["risks_json"]),
            invalid_conditions=cls._list(row["invalid_conditions_json"]),
            future_return_1d=row["future_return_1d"], future_return_3d=row["future_return_3d"],
            future_return_5d=row["future_return_5d"], future_return_10d=row["future_return_10d"],
            future_max_drawdown_5d=row["future_max_drawdown_5d"],
            future_max_drawdown_10d=row["future_max_drawdown_10d"],
            tracking_status=TrackingStatus(row["tracking_status"]),
            manual_verdict=ManualVerdict(row["manual_verdict"]) if row["manual_verdict"] else None,
            manual_notes=row["manual_notes"], created_at=cls._dt(row["created_at"]),
            updated_at=cls._dt(row["updated_at"]), metadata=cls._dict(row["metadata_json"]),
        )

    @staticmethod
    def _json(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, default=str)

    @staticmethod
    def _list(value: str | None) -> list:
        try:
            result = json.loads(value or "[]")
            return result if isinstance(result, list) else []
        except (TypeError, json.JSONDecodeError):
            return []

    @staticmethod
    def _dict(value: str | None) -> dict:
        try:
            result = json.loads(value or "{}")
            return result if isinstance(result, dict) else {}
        except (TypeError, json.JSONDecodeError):
            return {}

    @staticmethod
    def _dt(value: str) -> datetime:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
