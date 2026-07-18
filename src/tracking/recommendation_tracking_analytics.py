from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


HORIZONS = (1, 3, 5, 10)
GROUP_DIMENSIONS = (
    "action",
    "action_level",
    "source_type",
    "quant_decision",
    "ai_view",
    "manual_verdict",
    "tracking_status",
)
EXPECTED_COLUMNS = {
    "id", "symbol", "stock_name", "as_of_time", "source_type", "action",
    "action_level", "confidence", "final_score", "ai_view", "quant_decision",
    "current_price", "future_return_1d", "future_return_3d", "future_return_5d",
    "future_return_10d", "future_max_drawdown_5d", "tracking_status",
    "manual_verdict", "manual_notes", "reason_json", "risks_json", "metadata_json",
}
CASE_COLUMNS = [
    "case_type", "id", "as_of_time", "symbol", "stock_name", "source_type",
    "action", "action_level", "quant_decision", "confidence", "final_score",
    "current_price", "future_return_1d", "future_return_3d", "future_return_5d",
    "future_return_10d", "future_max_drawdown_5d", "tracking_status",
    "manual_verdict", "reason_summary", "risk_summary",
]
POSITIVE_ACTIONS = {"buy_candidate", "watch", "hold"}
NEGATIVE_ACTIONS = {"avoid", "risk_warning", "sell_alert", "stop_loss", "reduce"}
SUPPORT_DECISIONS = {"support"}
REJECT_DECISIONS = {"reject"}


def _clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _clean_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clean_json(item) for item in value]
    if isinstance(value, tuple):
        return [_clean_json(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return None if not np.isfinite(value) else value
    if value is not None and not isinstance(value, (str, bytes)):
        try:
            if pd.isna(value):
                return None
        except (TypeError, ValueError):
            pass
    return value


def _mean(values: pd.Series) -> float | None:
    numbers = pd.to_numeric(values, errors="coerce")
    result = numbers.mean()
    return None if pd.isna(result) else float(result)


def _median(values: pd.Series) -> float | None:
    numbers = pd.to_numeric(values, errors="coerce")
    result = numbers.median()
    return None if pd.isna(result) else float(result)


def _rate(mask: pd.Series) -> float | None:
    if len(mask) == 0:
        return None
    result = mask.mean()
    return None if pd.isna(result) else float(result)


def _safe_json_list(value: Any) -> list:
    try:
        parsed = json.loads(value or "[]")
    except (TypeError, json.JSONDecodeError):
        return []
    return parsed if isinstance(parsed, list) else []


class RecommendationTrackingAnalytics:
    def __init__(self, database_path: str = "data/signals.sqlite") -> None:
        self.database_path = Path(database_path)
        self.missing_columns: list[str] = []

    def build_report(self, output_dir: str = "data/tracking") -> dict[str, Any]:
        before = self._db_state()
        frame, actual_columns = self.load_tracking_frame()
        self.missing_columns = sorted(EXPECTED_COLUMNS.difference(actual_columns))
        frame = self._prepare_frame(frame)

        metrics_rows: list[dict[str, Any]] = []
        summary = self._build_summary(frame, metrics_rows)
        cases = self._build_cases(frame)
        summary["case_counts"] = cases["case_type"].value_counts().to_dict() if not cases.empty else {}
        summary["database_state_before"] = before

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        summary_path = output_path / "recommendation_tracking_summary.json"
        metrics_path = output_path / "recommendation_tracking_metrics.csv"
        cases_path = output_path / "recommendation_tracking_cases.csv"

        pd.DataFrame(metrics_rows, columns=[
            "analysis_type", "group_dimension", "group_value", "horizon",
            "metric", "value", "sample_count", "completed_count", "note",
        ]).to_csv(metrics_path, index=False, encoding="utf-8-sig")
        cases.to_csv(cases_path, index=False, encoding="utf-8-sig")

        after = self._db_state()
        summary["database_state_after"] = after
        summary["database_unchanged"] = (
            before.get("size") == after.get("size")
            and before.get("mtime") == after.get("mtime")
            and before.get("tracking_rows") == after.get("tracking_rows")
        )
        summary["outputs"] = {
            "summary": str(summary_path),
            "metrics": str(metrics_path),
            "cases": str(cases_path),
        }
        summary_path.write_text(
            json.dumps(_clean_json(summary), ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        return summary

    def load_tracking_frame(self) -> tuple[pd.DataFrame, set[str]]:
        if not self.database_path.exists():
            return pd.DataFrame(), set()
        uri = f"file:{self.database_path.resolve().as_posix()}?mode=ro"
        with sqlite3.connect(uri, uri=True) as connection:
            rows = connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='recommendation_tracking'"
            ).fetchall()
            if not rows:
                return pd.DataFrame(), set()
            info = connection.execute("PRAGMA table_info(recommendation_tracking)").fetchall()
            columns = [row[1] for row in info]
            frame = pd.read_sql_query(
                "SELECT * FROM recommendation_tracking",
                connection,
            )
        return frame, set(columns)

    def _db_state(self) -> dict[str, Any]:
        state: dict[str, Any] = {
            "path": str(self.database_path),
            "exists": self.database_path.exists(),
        }
        if not self.database_path.exists():
            return state
        stat = self.database_path.stat()
        state["size"] = int(stat.st_size)
        state["mtime"] = float(stat.st_mtime)
        try:
            uri = f"file:{self.database_path.resolve().as_posix()}?mode=ro"
            with sqlite3.connect(uri, uri=True) as connection:
                exists = connection.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name='recommendation_tracking'"
                ).fetchone()
                state["tracking_rows"] = (
                    int(connection.execute("SELECT COUNT(*) FROM recommendation_tracking").fetchone()[0])
                    if exists else 0
                )
        except sqlite3.Error as exc:
            state["error"] = f"{type(exc).__name__}: {exc}"
        return state

    def _prepare_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.copy()
        for column in EXPECTED_COLUMNS:
            if column not in result.columns:
                result[column] = np.nan
        result["as_of_date"] = pd.to_datetime(
            result["as_of_time"], errors="coerce"
        ).dt.strftime("%Y-%m-%d")
        for horizon in HORIZONS:
            column = f"future_return_{horizon}d"
            result[column] = pd.to_numeric(result[column], errors="coerce")
        result["future_max_drawdown_5d"] = pd.to_numeric(
            result["future_max_drawdown_5d"], errors="coerce"
        )
        result["confidence_normalized"] = pd.to_numeric(
            result["confidence"], errors="coerce"
        )
        high_confidence = result["confidence_normalized"].gt(1.5)
        result.loc[high_confidence, "confidence_normalized"] = (
            result.loc[high_confidence, "confidence_normalized"] / 100.0
        )
        result["final_score"] = pd.to_numeric(result["final_score"], errors="coerce")
        result["confidence_bucket"] = result["confidence_normalized"].apply(self._confidence_bucket)
        result["final_score_bucket"] = self._final_score_buckets(result["final_score"])
        result["action_quant_decision"] = (
            result["action"].fillna("missing").astype(str)
            + " + "
            + result["quant_decision"].fillna("missing").astype(str)
        )
        return result

    @staticmethod
    def _confidence_bucket(value: Any) -> str:
        if pd.isna(value):
            return "confidence_missing"
        value = float(value)
        if value < 0.40:
            return "confidence_lt_40"
        if value < 0.60:
            return "confidence_40_60"
        if value < 0.75:
            return "confidence_60_75"
        if value < 0.90:
            return "confidence_75_90"
        return "confidence_ge_90"

    @staticmethod
    def _final_score_buckets(scores: pd.Series) -> pd.Series:
        result = pd.Series("score_missing", index=scores.index, dtype="object")
        valid = pd.to_numeric(scores, errors="coerce")
        mask = valid.notna()
        if mask.sum() >= 5 and valid.loc[mask].nunique() >= 4:
            q20, q50, q80 = valid.loc[mask].quantile([0.2, 0.5, 0.8]).tolist()
            result.loc[mask & valid.le(q20)] = "bottom_20pct"
            result.loc[mask & valid.gt(q20) & valid.le(q50)] = "lower_middle_30pct"
            result.loc[mask & valid.gt(q50) & valid.lt(q80)] = "upper_middle_30pct"
            result.loc[mask & valid.ge(q80)] = "top_20pct"
        else:
            result.loc[mask & valid.lt(40)] = "bottom_20pct"
            result.loc[mask & valid.ge(40) & valid.lt(55)] = "lower_middle_30pct"
            result.loc[mask & valid.ge(55) & valid.lt(75)] = "upper_middle_30pct"
            result.loc[mask & valid.ge(75)] = "top_20pct"
        return result

    def _build_summary(self, frame: pd.DataFrame, metrics_rows: list[dict[str, Any]]) -> dict[str, Any]:
        summary: dict[str, Any] = {
            "research_only": True,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "database_path": str(self.database_path),
            "total_records": int(len(frame)),
            "date_range": self._date_range(frame),
            "completion_progress": self._completion_progress(frame),
            "overall_metrics": self._overall_metrics(frame, metrics_rows),
            "missing_columns": list(self.missing_columns),
            "limitations": [
                "当前是历史推荐复盘。",
                "未考虑真实成交、滑点和手续费。",
                "不同日期样本可能相关。",
                "样本量较小时结论不稳定。",
                "结果不构成投资建议。",
                "本统计不修改正式推荐系统。",
            ],
        }
        for dimension in GROUP_DIMENSIONS:
            key = f"by_{dimension}"
            if dimension in self.missing_columns:
                summary[key] = []
                continue
            summary[key] = self._group_summary(frame, dimension, metrics_rows)
        if "confidence" not in self.missing_columns:
            summary["by_confidence_bucket"] = self._group_summary(
                frame, "confidence_bucket", metrics_rows, "confidence_bucket"
            )
        else:
            summary["by_confidence_bucket"] = []
        if "final_score" not in self.missing_columns:
            summary["final_score_bucket_method"] = self._final_score_method(frame)
            summary["by_final_score_bucket"] = self._group_summary(
                frame, "final_score_bucket", metrics_rows, "final_score_bucket",
                include_score_stats=True,
            )
        else:
            summary["by_final_score_bucket"] = []
        if "action" not in self.missing_columns and "quant_decision" not in self.missing_columns:
            summary["by_action_quant_decision"] = self._group_summary(
                frame, "action_quant_decision", metrics_rows, "action_quant_decision"
            )
        else:
            summary["by_action_quant_decision"] = []
        summary["by_as_of_date"] = self._as_of_date_summary(frame, metrics_rows)
        summary["diagnostic_hints"] = self._diagnostic_hints(summary)
        return summary

    @staticmethod
    def _date_range(frame: pd.DataFrame) -> dict[str, Any]:
        dates = pd.to_datetime(frame.get("as_of_time"), errors="coerce")
        if dates.notna().any():
            return {
                "min": dates.min().isoformat(),
                "max": dates.max().isoformat(),
            }
        return {"min": None, "max": None}

    def _completion_progress(self, frame: pd.DataFrame) -> dict[str, Any]:
        result = {"total_records": int(len(frame))}
        for horizon in HORIZONS:
            completed = int(frame[f"future_return_{horizon}d"].notna().sum())
            result[f"completed_{horizon}d_count"] = completed
            result[f"pending_{horizon}d_count"] = int(len(frame) - completed)
        for status in ("complete", "tracking", "failed"):
            if "tracking_status" in frame:
                result[f"{status}_tracking_count"] = int(
                    frame["tracking_status"].fillna("missing").astype(str).eq(status).sum()
                )
        result["pending_tracking_count"] = result.get("tracking_tracking_count", 0)
        return result

    def _overall_metrics(self, frame: pd.DataFrame, metrics_rows: list[dict[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for horizon in HORIZONS:
            metrics = self._horizon_metrics(frame, horizon)
            result[f"{horizon}d"] = metrics
            self._append_metrics(metrics_rows, "overall", "all", "all", horizon, metrics)
        drawdown = pd.to_numeric(frame["future_max_drawdown_5d"], errors="coerce")
        valid = drawdown.dropna()
        result["drawdown_5d"] = {
            "sample_count": int(valid.shape[0]),
            "avg_max_drawdown_5d": _mean(drawdown),
            "median_max_drawdown_5d": _median(drawdown),
            "drawdown_below_minus_5pct_rate": _rate(valid.le(-0.05)) if len(valid) else None,
            "drawdown_below_minus_10pct_rate": _rate(valid.le(-0.10)) if len(valid) else None,
        }
        return result

    def _horizon_metrics(self, frame: pd.DataFrame, horizon: int) -> dict[str, Any]:
        column = f"future_return_{horizon}d"
        values = pd.to_numeric(frame[column], errors="coerce")
        valid = values.dropna()
        return {
            "sample_count": int(len(valid)),
            "avg_return": _mean(values),
            "median_return": _median(values),
            "positive_return_rate": _rate(valid.gt(0)) if len(valid) else None,
            "negative_return_rate": _rate(valid.lt(0)) if len(valid) else None,
            "best_return": None if valid.empty else float(valid.max()),
            "worst_return": None if valid.empty else float(valid.min()),
        }

    def _group_summary(
        self,
        frame: pd.DataFrame,
        dimension: str,
        metrics_rows: list[dict[str, Any]],
        group_dimension_name: str | None = None,
        include_score_stats: bool = False,
    ) -> list[dict[str, Any]]:
        if dimension not in frame.columns:
            return []
        rows: list[dict[str, Any]] = []
        for value, group in frame.groupby(dimension, dropna=False, sort=True):
            group_value = "missing" if pd.isna(value) else str(value)
            item: dict[str, Any] = {
                "group": group_value,
                "total_records": int(len(group)),
                "avg_max_drawdown_5d": _mean(group["future_max_drawdown_5d"]),
            }
            for horizon in HORIZONS:
                completed = group[f"future_return_{horizon}d"].notna()
                item[f"completed_{horizon}d_count"] = int(completed.sum())
                item[f"avg_return_{horizon}d"] = _mean(group[f"future_return_{horizon}d"])
                item[f"positive_rate_{horizon}d"] = self._positive_rate(group, horizon)
            if include_score_stats:
                scores = pd.to_numeric(group["final_score"], errors="coerce")
                item["score_min"] = None if scores.dropna().empty else float(scores.min())
                item["score_max"] = None if scores.dropna().empty else float(scores.max())
                item["avg_score"] = _mean(scores)
            rows.append(item)
            self._append_group_rows(
                metrics_rows, group_dimension_name or dimension, group_value, item
            )
        return rows

    @staticmethod
    def _positive_rate(frame: pd.DataFrame, horizon: int) -> float | None:
        values = pd.to_numeric(frame[f"future_return_{horizon}d"], errors="coerce").dropna()
        return _rate(values.gt(0)) if len(values) else None

    def _append_group_rows(
        self,
        metrics_rows: list[dict[str, Any]],
        dimension: str,
        group_value: str,
        item: dict[str, Any],
    ) -> None:
        for horizon in HORIZONS:
            sample_count = int(item.get("total_records") or 0)
            completed = int(item.get(f"completed_{horizon}d_count") or 0)
            for metric in ("avg_return", "positive_rate"):
                value = item.get(f"{metric}_{horizon}d")
                metrics_rows.append({
                    "analysis_type": "group_performance",
                    "group_dimension": dimension,
                    "group_value": group_value,
                    "horizon": f"{horizon}d",
                    "metric": metric,
                    "value": value,
                    "sample_count": sample_count,
                    "completed_count": completed,
                    "note": None,
                })
        metrics_rows.append({
            "analysis_type": "group_performance",
            "group_dimension": dimension,
            "group_value": group_value,
            "horizon": "5d",
            "metric": "avg_max_drawdown",
            "value": item.get("avg_max_drawdown_5d"),
            "sample_count": int(item.get("total_records") or 0),
            "completed_count": int(item.get("completed_5d_count") or 0),
            "note": None,
        })

    def _append_metrics(
        self,
        metrics_rows: list[dict[str, Any]],
        analysis_type: str,
        dimension: str,
        group_value: str,
        horizon: int,
        metrics: dict[str, Any],
    ) -> None:
        for metric, value in metrics.items():
            metrics_rows.append({
                "analysis_type": analysis_type,
                "group_dimension": dimension,
                "group_value": group_value,
                "horizon": f"{horizon}d",
                "metric": metric,
                "value": value,
                "sample_count": metrics.get("sample_count"),
                "completed_count": metrics.get("sample_count"),
                "note": None,
            })

    @staticmethod
    def _final_score_method(frame: pd.DataFrame) -> str:
        valid = pd.to_numeric(frame["final_score"], errors="coerce").dropna()
        return "quantile" if len(valid) >= 5 and valid.nunique() >= 4 else "fixed_range_fallback"

    def _as_of_date_summary(self, frame: pd.DataFrame, metrics_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        rows = self._group_summary(frame, "as_of_date", metrics_rows, "as_of_date")
        for row in rows:
            if row.get("completed_5d_count", 0) < 10:
                row["sample_note"] = "small_sample"
        return rows

    def _build_cases(self, frame: pd.DataFrame) -> pd.DataFrame:
        cases: list[pd.DataFrame] = []
        completed_5d = frame["future_return_5d"].notna()
        positive_action = frame["action"].fillna("").astype(str).isin(POSITIVE_ACTIONS)
        negative_action = frame["action"].fillna("").astype(str).isin(NEGATIVE_ACTIONS)
        high_confidence = frame["confidence_normalized"].ge(0.75)
        support = frame["quant_decision"].fillna("").astype(str).isin(SUPPORT_DECISIONS)
        reject = frame["quant_decision"].fillna("").astype(str).isin(REJECT_DECISIONS)
        top_score = frame["final_score_bucket"].eq("top_20pct")
        bottom_score = frame["final_score_bucket"].eq("bottom_20pct")

        selectors = [
            ("positive_action_large_loss", positive_action & completed_5d & frame["future_return_5d"].le(-0.05), "future_return_5d", True),
            ("positive_action_large_drawdown", positive_action & frame["future_max_drawdown_5d"].le(-0.08), "future_max_drawdown_5d", True),
            ("negative_action_large_gain", negative_action & completed_5d & frame["future_return_5d"].ge(0.05), "future_return_5d", False),
            ("high_confidence_failure", high_confidence & completed_5d & frame["future_return_5d"].le(-0.05), "future_return_5d", True),
            ("high_score_failure", top_score & completed_5d & frame["future_return_5d"].le(-0.05), "future_return_5d", True),
            ("low_score_success", bottom_score & completed_5d & frame["future_return_5d"].ge(0.05), "future_return_5d", False),
            ("rule_support_failure", support & completed_5d & frame["future_return_5d"].le(-0.05), "future_return_5d", True),
            ("rule_reject_success", reject & completed_5d & frame["future_return_5d"].ge(0.05), "future_return_5d", False),
            ("best_cases", completed_5d, "future_return_5d", False),
            ("worst_cases", completed_5d, "future_return_5d", True),
        ]
        for case_type, mask, sort_column, ascending in selectors:
            subset = frame.loc[mask].sort_values(sort_column, ascending=ascending).head(20).copy()
            if subset.empty:
                continue
            subset["case_type"] = case_type
            cases.append(self._case_frame(subset))
        if not cases:
            return pd.DataFrame(columns=CASE_COLUMNS)
        return pd.concat(cases, ignore_index=True).reindex(columns=CASE_COLUMNS)

    def _case_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.copy()
        result["reason_summary"] = result["reason_json"].apply(
            lambda value: "；".join(str(item) for item in _safe_json_list(value)[:2])
        )
        result["risk_summary"] = result["risks_json"].apply(
            lambda value: "；".join(str(item) for item in _safe_json_list(value)[:2])
        )
        for column in CASE_COLUMNS:
            if column not in result.columns:
                result[column] = None
        return result[CASE_COLUMNS]

    @staticmethod
    def _diagnostic_hints(summary: dict[str, Any]) -> list[str]:
        hints: list[str] = []
        progress = summary.get("completion_progress", {})
        if progress.get("completed_5d_count", 0) < 30:
            hints.append("insufficient_completed_5d_samples")
        if summary.get("total_records", 0) < 50:
            hints.append("sample_size_too_small")
        action_groups = {item["group"]: item for item in summary.get("by_action", [])}
        buy = action_groups.get("buy_candidate")
        avoid = action_groups.get("avoid")
        if buy and avoid:
            if (buy.get("avg_return_5d") is not None and avoid.get("avg_return_5d") is not None
                    and buy["avg_return_5d"] < avoid["avg_return_5d"]):
                hints.append("positive_action_underperformed_negative_action")
        quant_groups = {item["group"]: item for item in summary.get("by_quant_decision", [])}
        support = quant_groups.get("support")
        reject = quant_groups.get("reject")
        if support and reject:
            if (support.get("avg_return_5d") is not None and reject.get("avg_return_5d") is not None
                    and support["avg_return_5d"] < reject["avg_return_5d"]):
                hints.append("support_group_underperformed")
        date_rows = summary.get("by_as_of_date", [])
        returns = [
            item.get("avg_return_5d") for item in date_rows
            if item.get("avg_return_5d") is not None and item.get("completed_5d_count", 0) >= 10
        ]
        if len(returns) >= 2 and max(returns) - min(returns) > 0.10:
            hints.append("strong_date_dependence")
        return hints
