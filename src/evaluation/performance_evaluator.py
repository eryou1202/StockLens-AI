from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


class PerformanceEvaluator:
    def __init__(self, dataset_path: str = "data/ml_dataset.csv"):
        self.dataset_path = Path(dataset_path)

    def summarize(self) -> dict[str, Any]:
        if not self.dataset_path.exists():
            raise FileNotFoundError(str(self.dataset_path))
        frame = pd.read_csv(self.dataset_path)
        status = self._text_column(frame, "feedback_status").fillna("pending")
        complete_count = int((status == "complete").sum())

        summary: dict[str, Any] = {
            "total_samples": int(len(frame)),
            "complete_samples": complete_count,
            "partial_samples": int((status == "partial").sum()),
            "pending_samples": int((status == "pending").sum()),
            "failed_samples": int((status == "failed").sum()),
            "overall": self._stats(frame),
            "by_quant_decision": self._group_stats(
                frame, "quant_decision", ["support", "uncertain", "reject"]
            ),
            "by_final_level": self._group_stats(
                frame, "final_level", ["strong_watch", "watch", "risky", "avoid"]
            ),
            "by_ai_view": self._group_stats(
                frame, "ai_view", ["positive", "neutral", "negative", "uncertain"]
            ),
            "by_quant_score_bucket": self._quant_score_buckets(frame),
            "sample_warning": complete_count < 5,
        }
        return summary

    def print_summary(self) -> None:
        summary = self.summarize()
        print("StockLens Feedback Summary\n")
        print(f"Total samples: {summary['total_samples']}")
        print(f"Complete labels: {summary['complete_samples']}")
        print(f"Partial labels: {summary['partial_samples']}")
        print(f"Pending labels: {summary['pending_samples']}")
        print(f"Failed labels: {summary['failed_samples']}")

        print("\nOverall:")
        self._print_stats(summary["overall"])
        self._print_group("By quant_decision", summary["by_quant_decision"])
        self._print_group("By final_level", summary["by_final_level"])
        self._print_group("By ai_view", summary["by_ai_view"])
        self._print_group("By quant_score bucket", summary["by_quant_score_bucket"])

        if summary["sample_warning"]:
            print("\n当前完整反馈样本太少，统计结果仅用于检查流程，不代表策略有效性。")

    @classmethod
    def _stats(cls, frame: pd.DataFrame) -> dict[str, Any]:
        returns = cls._numeric_column(frame, "future_return_5d")
        drawdowns = cls._numeric_column(frame, "future_max_drawdown_5d")
        hits = cls._hit_column(frame, "hit_5d")
        return {
            "count": int(len(frame)),
            "labeled_5d_count": int(returns.notna().sum()),
            "avg_future_return_5d": cls._mean_or_none(returns),
            "hit_rate_5d": cls._mean_or_none(hits),
            "avg_future_max_drawdown_5d": cls._mean_or_none(drawdowns),
        }

    @classmethod
    def _group_stats(
        cls,
        frame: pd.DataFrame,
        column: str,
        expected_values: list[str],
    ) -> dict[str, dict[str, Any]]:
        values = cls._text_column(frame, column)
        return {
            value: cls._stats(frame.loc[values == value])
            for value in expected_values
        }

    @classmethod
    def _quant_score_buckets(cls, frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
        labels = ["0-20", "20-40", "40-60", "60-80", "80-100"]
        scores = cls._numeric_column(frame, "quant_score")
        buckets = pd.cut(
            scores,
            bins=[0, 20, 40, 60, 80, 100.0000001],
            labels=labels,
            include_lowest=True,
            right=False,
        )
        return {label: cls._stats(frame.loc[buckets == label]) for label in labels}

    @staticmethod
    def _numeric_column(frame: pd.DataFrame, column: str) -> pd.Series:
        if column not in frame:
            return pd.Series(float("nan"), index=frame.index, dtype="float64")
        return pd.to_numeric(frame[column], errors="coerce")

    @staticmethod
    def _text_column(frame: pd.DataFrame, column: str) -> pd.Series:
        if column not in frame:
            return pd.Series(None, index=frame.index, dtype="object")
        return frame[column].astype("string").str.strip().str.lower()

    @staticmethod
    def _hit_column(frame: pd.DataFrame, column: str) -> pd.Series:
        if column not in frame:
            return pd.Series(float("nan"), index=frame.index, dtype="float64")
        mapping = {
            "true": 1.0, "1": 1.0, "1.0": 1.0, "yes": 1.0,
            "false": 0.0, "0": 0.0, "0.0": 0.0, "no": 0.0,
        }
        return frame[column].astype("string").str.strip().str.lower().map(mapping).astype("float64")

    @staticmethod
    def _mean_or_none(series: pd.Series) -> float | None:
        clean = series.dropna()
        return None if clean.empty else float(clean.mean())

    @classmethod
    def _print_group(cls, title: str, groups: dict[str, dict[str, Any]]) -> None:
        print(f"\n{title}:")
        for name, stats in groups.items():
            print(f"  {name}: ", end="")
            cls._print_stats(stats, indent="")

    @staticmethod
    def _print_stats(stats: dict[str, Any], indent: str = "  ") -> None:
        def fmt(value: float | None, percent: bool = False) -> str:
            if value is None:
                return "-"
            return f"{value:.2%}" if percent else f"{value:.6f}"

        print(
            f"{indent}count={stats['count']}, labeled_5d={stats['labeled_5d_count']}, "
            f"avg_return_5d={fmt(stats['avg_future_return_5d'], True)}, "
            f"hit_rate_5d={fmt(stats['hit_rate_5d'], True)}, "
            f"avg_drawdown_5d={fmt(stats['avg_future_max_drawdown_5d'], True)}"
        )
