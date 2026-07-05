from __future__ import annotations

import json
import math
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd

from src.storage.sqlite_store import SQLiteSignalStore


class MLDatasetBuilder:
    BASE_COLUMNS = [
        "symbol", "stock_name", "as_of_time", "final_level", "final_score", "ai_view", "quant_view",
    ]
    AI_COLUMNS = [
        "ai_sentiment_score", "ai_event_strength", "ai_source_confidence", "ai_confidence",
        "ai_overall_score", "event_type", "source_tier_avg", "risk_flag_count", "contradiction_count",
    ]
    QUANT_FEATURE_COLUMNS = [
        "return_1d", "return_3d", "return_5d", "return_10d", "return_20d", "return_60d",
        "ma5_ma20_gap", "ma20_ma60_gap", "close_ma20_gap",
        "volume_ratio_5d", "volume_ratio_20d", "amount_ratio_5d", "amount_ratio_20d",
        "volatility_20d", "max_drawdown_20d", "atr_14", "rsi_14", "macd_hist",
        "bollinger_position", "data_quality_score",
    ]
    QUANT_SCORE_COLUMNS = [
        "trend_score", "momentum_score", "volume_score", "risk_score", "overheat_score",
        "macd_score", "quant_score", "heuristic_prob_up_5d", "quant_decision",
    ]
    LABEL_COLUMNS = [
        "future_return_1d", "future_return_3d", "future_return_5d", "future_return_10d",
        "future_max_drawdown_5d", "future_max_drawdown_10d",
        "hit_1d", "hit_3d", "hit_5d", "hit_10d", "feedback_status",
    ]
    OUTPUT_COLUMNS = BASE_COLUMNS + AI_COLUMNS + QUANT_FEATURE_COLUMNS + QUANT_SCORE_COLUMNS + LABEL_COLUMNS

    def __init__(self, database_path: str):
        self.database_path = Path(database_path)
        SQLiteSignalStore(str(self.database_path))  # 自动建表/迁移，也覆盖数据库不存在的情况。

    def build(self, output_path: str = "data/ml_dataset.csv") -> pd.DataFrame:
        rows = self._read_joined_rows()
        records = [self._to_record(row) for row in rows]
        frame = pd.DataFrame(records, columns=self.OUTPUT_COLUMNS)

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False, encoding="utf-8-sig")
        return frame

    def _read_joined_rows(self) -> list[sqlite3.Row]:
        with sqlite3.connect(self.database_path) as conn:
            conn.row_factory = sqlite3.Row
            return conn.execute(
                """
                SELECT s.id AS snapshot_id, s.symbol AS snapshot_symbol,
                       s.as_of_time AS snapshot_as_of_time,
                       s.final_level AS snapshot_final_level,
                       s.final_score AS snapshot_final_score,
                       s.final_decision_json,
                       f.future_return_1d, f.future_return_3d,
                       f.future_return_5d, f.future_return_10d,
                       f.future_max_drawdown_5d, f.future_max_drawdown_10d,
                       f.hit_1d, f.hit_3d, f.hit_5d, f.hit_10d,
                       f.feedback_status
                FROM signal_snapshots AS s
                LEFT JOIN signal_feedback AS f
                  ON f.symbol = s.symbol AND f.as_of_time = s.as_of_time
                WHERE s.id = (
                    SELECT MAX(s2.id)
                    FROM signal_snapshots AS s2
                    WHERE s2.symbol = s.symbol AND s2.as_of_time = s.as_of_time
                )
                ORDER BY s.as_of_time ASC, s.id ASC
                """
            ).fetchall()

    def _to_record(self, row: sqlite3.Row) -> dict[str, Any]:
        payload = self._json_object(row["final_decision_json"])
        ai = payload.get("ai_candidate") or {}
        quant = payload.get("quant_result") or {}
        legacy_features = quant.get("features") or {}
        extra = legacy_features.get("extra") or {}
        feature_vector = extra.get("feature_vector") or {}
        score_breakdown = extra.get("score_breakdown") or {}
        ai_metadata = ai.get("metadata") or {}
        signal_candidate = ai_metadata.get("signal_candidate") or {}

        sources = ai_metadata.get("sources") or signal_candidate.get("sources") or []
        tiers = [self._number(item.get("source_tier")) for item in sources if isinstance(item, dict)]
        tiers = [value for value in tiers if value is not None]
        risk_flags = ai.get("risk_flags") or ai_metadata.get("risk_flags") or signal_candidate.get("risk_flags") or []
        contradictions = ai_metadata.get("contradictions") or signal_candidate.get("contradictions") or []

        record: dict[str, Any] = {
            "symbol": payload.get("symbol") or row["snapshot_symbol"],
            "stock_name": payload.get("stock_name") or ai.get("stock_name"),
            "as_of_time": payload.get("as_of_time") or row["snapshot_as_of_time"],
            "final_level": payload.get("final_level") or row["snapshot_final_level"],
            "final_score": payload.get("final_score", row["snapshot_final_score"]),
            "ai_view": payload.get("ai_view"),
            "quant_view": payload.get("quant_view"),
            "ai_sentiment_score": ai.get("sentiment_score"),
            "ai_event_strength": ai.get("event_strength"),
            "ai_source_confidence": ai.get("source_confidence"),
            "ai_confidence": ai.get("ai_confidence"),
            "ai_overall_score": ai_metadata.get("ai_overall_score", signal_candidate.get("ai_overall_score")),
            "event_type": ai.get("event_type"),
            "source_tier_avg": sum(tiers) / len(tiers) if tiers else None,
            "risk_flag_count": len(risk_flags) if isinstance(risk_flags, list) else 0,
            "contradiction_count": len(contradictions) if isinstance(contradictions, list) else 0,
        }

        for column in self.QUANT_FEATURE_COLUMNS:
            record[column] = feature_vector.get(column, legacy_features.get(column))

        for column in self.QUANT_SCORE_COLUMNS:
            if column == "heuristic_prob_up_5d":
                record[column] = score_breakdown.get(column, quant.get("prob_up_5d"))
            elif column == "quant_decision":
                value = extra.get("internal_quant_decision") or score_breakdown.get(column)
                record[column] = value or self._legacy_quant_decision(quant.get("quant_decision"))
            else:
                record[column] = score_breakdown.get(column, quant.get(column))

        for column in self.LABEL_COLUMNS:
            if column == "feedback_status":
                record[column] = row[column] or "pending"
            elif column.startswith("hit_"):
                record[column] = self._sqlite_bool(row[column])
            else:
                record[column] = row[column]
        return record

    @staticmethod
    def _json_object(value: Any) -> dict[str, Any]:
        if isinstance(value, dict):
            return value
        try:
            parsed = json.loads(value or "{}")
        except (TypeError, json.JSONDecodeError):
            return {}
        return parsed if isinstance(parsed, dict) else {}

    @staticmethod
    def _number(value: Any) -> float | None:
        try:
            result = float(value)
        except (TypeError, ValueError, OverflowError):
            return None
        return result if math.isfinite(result) else None

    @staticmethod
    def _sqlite_bool(value: Any) -> bool | None:
        if value is None:
            return None
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"true", "1", "yes"}:
                return True
            if normalized in {"false", "0", "no"}:
                return False
            return None
        return bool(value)

    @staticmethod
    def _legacy_quant_decision(value: Any) -> str | None:
        if value in {"strong_watch", "watch", "support"}:
            return "support"
        if value in {"risky", "uncertain", "neutral"}:
            return "uncertain"
        if value in {"avoid", "reject"}:
            return "reject"
        return None
