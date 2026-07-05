from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


class ModelDiagnostics:
    FEATURE_COLUMNS = [
        "return_1d", "return_3d", "return_5d", "return_10d", "return_20d", "return_60d",
        "ma5_ma20_gap", "ma20_ma60_gap", "close_ma20_gap",
        "volume_ratio_5d", "volume_ratio_20d", "amount_ratio_5d", "amount_ratio_20d",
        "volatility_20d", "max_drawdown_20d", "atr_14", "rsi_14", "macd_hist",
        "bollinger_position", "trend_score", "momentum_score", "volume_score", "risk_score",
        "overheat_score", "macd_score", "quant_score", "heuristic_prob_up_5d",
    ]

    def __init__(self, dataset_path: str = "data/ml_dataset.csv"):
        self.dataset_path = Path(dataset_path)

    def summarize_dataset(self) -> dict[str, Any]:
        frame = self._load()
        status = self._text_column(frame, "feedback_status", lower=True).fillna("pending")
        complete = frame.loc[status == "complete"].copy()
        hits = self._hit_column(complete, "hit_5d")
        future_returns = self._numeric_column(complete, "future_return_5d")

        return {
            "rows": int(len(frame)),
            "complete_rows": int((status == "complete").sum()),
            "pending_rows": int((status == "pending").sum()),
            "partial_rows": int((status == "partial").sum()),
            "hit_5d_distribution": {
                "0": int((hits == 0).sum()),
                "1": int((hits == 1).sum()),
                "missing": int(hits.isna().sum()),
            },
            "future_return_5d": self._describe(future_returns),
            "by_symbol": self._group_stats(complete, "symbol"),
            "by_quant_decision": self._group_stats(
                complete,
                "quant_decision",
                expected=["support", "uncertain", "reject"],
                lower=True,
            ),
            "by_final_level": self._group_stats(
                complete,
                "final_level",
                expected=["strong_watch", "watch", "risky", "avoid"],
                lower=True,
            ),
            "by_quant_score_bucket": self._quant_score_buckets(complete),
        }

    def feature_target_correlation(
        self,
        target: str = "future_return_5d",
    ) -> pd.DataFrame:
        frame = self._load()
        status = self._text_column(frame, "feedback_status", lower=True)
        complete = frame.loc[status == "complete"].copy()
        target_values = self._numeric_column(complete, target)
        records: list[dict[str, Any]] = []

        for feature in self.FEATURE_COLUMNS:
            values = self._numeric_column(complete, feature)
            paired = pd.DataFrame({"feature": values, "target": target_values}).dropna()
            correlation = float("nan")
            if (
                len(paired) >= 3
                and paired["feature"].nunique() > 1
                and paired["target"].nunique() > 1
            ):
                correlation = float(paired["feature"].corr(paired["target"], method="pearson"))
            records.append(
                {
                    "feature": feature,
                    "correlation": correlation,
                    "abs_correlation": abs(correlation),
                    "paired_samples": int(len(paired)),
                }
            )

        return pd.DataFrame(records).sort_values(
            "abs_correlation",
            ascending=False,
            na_position="last",
            kind="stable",
        ).reset_index(drop=True)

    def print_report(self) -> None:
        summary = self.summarize_dataset()
        print("StockLens ML Dataset Diagnostics\n")
        print(f"Rows: {summary['rows']}")
        print(f"Complete rows: {summary['complete_rows']}")
        print(f"Pending rows: {summary['pending_rows']}")
        print(f"Partial rows: {summary['partial_rows']}")

        distribution = summary["hit_5d_distribution"]
        print("\nTarget distribution hit_5d:")
        print(f"  0: {distribution['0']}")
        print(f"  1: {distribution['1']}")
        print(f"  missing: {distribution['missing']}")

        print("\nFuture return 5d:")
        for key, value in summary["future_return_5d"].items():
            print(f"  {key}: {self._format_number(value, percent=True)}")

        self._print_groups("By symbol", summary["by_symbol"])
        self._print_groups("By quant_decision", summary["by_quant_decision"])
        self._print_groups("By final_level", summary["by_final_level"])
        self._print_groups("By quant_score bucket", summary["by_quant_score_bucket"])

        correlations = self.feature_target_correlation("future_return_5d")
        valid = correlations.dropna(subset=["correlation"]).head(20)
        print("\nTop feature correlations with future_return_5d:")
        if valid.empty:
            print("  有效配对样本不足或特征没有变化，暂时无法计算相关性。")
        else:
            for row in valid.itertuples(index=False):
                print(
                    f"  {row.feature}: correlation={row.correlation:.6f}, "
                    f"paired_samples={row.paired_samples}"
                )

        if summary["complete_rows"] < 30:
            print("\n完整样本少于 30，分组和相关性结果只适合检查流程。")
        print("\n当前结果来自测试型历史样本，不代表真实历史回测表现。")

    def _load(self) -> pd.DataFrame:
        if not self.dataset_path.exists():
            raise FileNotFoundError(str(self.dataset_path))
        return pd.read_csv(self.dataset_path)

    @classmethod
    def _group_stats(
        cls,
        frame: pd.DataFrame,
        column: str,
        expected: list[str] | None = None,
        lower: bool = False,
    ) -> dict[str, dict[str, Any]]:
        values = cls._text_column(frame, column, lower=lower)
        if expected is None:
            groups = sorted(str(value) for value in values.dropna().unique())
        else:
            groups = expected
        return {group: cls._stats(frame.loc[values == group]) for group in groups}

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

    @classmethod
    def _stats(cls, frame: pd.DataFrame) -> dict[str, Any]:
        returns = cls._numeric_column(frame, "future_return_5d")
        hits = cls._hit_column(frame, "hit_5d")
        return {
            "count": int(len(frame)),
            "avg_future_return_5d": cls._mean_or_none(returns),
            "hit_rate_5d": cls._mean_or_none(hits),
        }

    @staticmethod
    def _describe(series: pd.Series) -> dict[str, float | None]:
        clean = series.dropna()
        if clean.empty:
            return {key: None for key in ("mean", "median", "std", "min", "max", "25%", "75%")}
        return {
            "mean": float(clean.mean()),
            "median": float(clean.median()),
            "std": float(clean.std()) if len(clean) > 1 else None,
            "min": float(clean.min()),
            "max": float(clean.max()),
            "25%": float(clean.quantile(0.25)),
            "75%": float(clean.quantile(0.75)),
        }

    @staticmethod
    def _numeric_column(frame: pd.DataFrame, column: str) -> pd.Series:
        if column not in frame:
            return pd.Series(float("nan"), index=frame.index, dtype="float64")
        return pd.to_numeric(frame[column], errors="coerce")

    @staticmethod
    def _text_column(frame: pd.DataFrame, column: str, lower: bool = False) -> pd.Series:
        if column not in frame:
            return pd.Series(None, index=frame.index, dtype="object")
        result = frame[column].astype("string").str.strip()
        return result.str.lower() if lower else result

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
    def _print_groups(cls, title: str, groups: dict[str, dict[str, Any]]) -> None:
        print(f"\n{title}:")
        for name, stats in groups.items():
            print(
                f"  {name}: count={stats['count']}, "
                f"avg_return_5d={cls._format_number(stats['avg_future_return_5d'], True)}, "
                f"hit_rate_5d={cls._format_number(stats['hit_rate_5d'], True)}"
            )

    @staticmethod
    def _format_number(value: float | None, percent: bool = False) -> str:
        if value is None or pd.isna(value):
            return "-"
        return f"{value:.2%}" if percent else f"{value:.6f}"
