from __future__ import annotations

import math
import re
from typing import Any

import numpy as np
import pandas as pd


class MLEvaluator:
    def evaluate_classification(
        self,
        frame: pd.DataFrame,
        target: str,
        probabilities: np.ndarray,
        predictions: np.ndarray,
    ) -> dict[str, Any]:
        from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

        truth = pd.to_numeric(frame[target], errors="coerce").astype(int).to_numpy()
        horizon = self._horizon(target)
        return_column = f"future_return_{horizon}d" if horizon else None
        returns = self._optional_numeric(frame, return_column)
        metrics: dict[str, Any] = {
            "task_type": "classification",
            "samples": int(len(frame)),
            "accuracy": float(accuracy_score(truth, predictions)),
            "precision": float(precision_score(truth, predictions, zero_division=0)),
            "recall": float(recall_score(truth, predictions, zero_division=0)),
            "positive_rate": float(np.mean(truth)),
            "hit_rate": float(np.mean(truth)),
            "pred_positive_rate": float(np.mean(predictions)),
            "roc_auc": None,
        }
        if len(np.unique(truth)) > 1:
            metrics["roc_auc"] = float(roc_auc_score(truth, probabilities))
        metrics.update(self._ranking(probabilities, returns, frame, horizon))
        metrics["mean_future_return_by_pred_bucket"] = metrics.get("buckets", [])
        return metrics

    def evaluate_regression(
        self,
        frame: pd.DataFrame,
        target: str,
        predictions: np.ndarray,
    ) -> dict[str, Any]:
        from sklearn.metrics import mean_absolute_error, mean_squared_error

        truth = pd.to_numeric(frame[target], errors="coerce").to_numpy(dtype=float)
        horizon = self._horizon(target)
        return_column = f"future_return_{horizon}d" if horizon else None
        returns = self._optional_numeric(frame, return_column)
        metrics: dict[str, Any] = {
            "task_type": "regression",
            "samples": int(len(frame)),
            "mse": float(mean_squared_error(truth, predictions)),
            "mae": float(mean_absolute_error(truth, predictions)),
            "pearson_corr": self._correlation(predictions, truth, "pearson"),
            "spearman_corr": self._correlation(predictions, truth, "spearman"),
        }
        metrics.update(self._ranking(predictions, returns, frame, horizon))
        return metrics

    def _ranking(
        self,
        scores: np.ndarray,
        returns: pd.Series | None,
        frame: pd.DataFrame,
        horizon: int | None,
    ) -> dict[str, Any]:
        if returns is None:
            return {
                "pearson_corr_with_future_return": None,
                "spearman_corr_with_future_return": None,
                "top20_avg_return": None,
                "middle60_avg_return": None,
                "bottom20_avg_return": None,
                "top_bottom_spread": None,
                "top20_avg_excess_return": None,
                "middle60_avg_excess_return": None,
                "bottom20_avg_excess_return": None,
                "top_bottom_excess_spread": None,
                "buckets": [],
            }
        excess_column = f"future_excess_return_{horizon}d" if horizon else None
        excess_returns = self._optional_numeric(frame, excess_column)
        work = pd.DataFrame({"score": scores, "future_return": returns.to_numpy()}, index=frame.index)
        work["future_excess_return"] = (
            excess_returns.to_numpy() if excess_returns is not None else np.nan
        )
        work = work.dropna(subset=["score", "future_return"]).sort_values("score", kind="stable")
        if work.empty:
            return self._ranking(np.array([]), None, frame, horizon)
        count = len(work)
        edge = max(1, int(math.floor(count * 0.20)))
        if edge * 2 >= count:
            edge = max(1, count // 3)
        work["bucket"] = "middle60"
        work.iloc[:edge, work.columns.get_loc("bucket")] = "bottom20"
        work.iloc[count - edge :, work.columns.get_loc("bucket")] = "top20"

        drawdown_column = f"future_max_drawdown_{horizon}d" if horizon else None
        if drawdown_column and drawdown_column in frame.columns:
            work["max_drawdown"] = pd.to_numeric(frame.loc[work.index, drawdown_column], errors="coerce")
        else:
            work["max_drawdown"] = np.nan

        buckets = []
        for name in ("top20", "middle60", "bottom20"):
            group = work.loc[work["bucket"] == name]
            buckets.append({
                "bucket": name,
                "count": int(len(group)),
                "avg_future_return": self._mean(group["future_return"]),
                "avg_future_excess_return": self._mean(group["future_excess_return"]),
                "median_future_return": self._median(group["future_return"]),
                "hit_rate": self._mean((group["future_return"] > 0).astype(float)),
                "avg_max_drawdown": self._mean(group["max_drawdown"]),
            })
        by_name = {item["bucket"]: item for item in buckets}
        top = by_name["top20"]["avg_future_return"]
        bottom = by_name["bottom20"]["avg_future_return"]
        top_excess = by_name["top20"]["avg_future_excess_return"]
        bottom_excess = by_name["bottom20"]["avg_future_excess_return"]
        return {
            "pearson_corr_with_future_return": self._correlation(
                work["score"].to_numpy(), work["future_return"].to_numpy(), "pearson"
            ),
            "spearman_corr_with_future_return": self._correlation(
                work["score"].to_numpy(), work["future_return"].to_numpy(), "spearman"
            ),
            "top20_avg_return": top,
            "middle60_avg_return": by_name["middle60"]["avg_future_return"],
            "bottom20_avg_return": bottom,
            "top_bottom_spread": None if top is None or bottom is None else top - bottom,
            "top20_avg_excess_return": top_excess,
            "middle60_avg_excess_return": by_name["middle60"]["avg_future_excess_return"],
            "bottom20_avg_excess_return": bottom_excess,
            "top_bottom_excess_spread": (
                None if top_excess is None or bottom_excess is None
                else top_excess - bottom_excess
            ),
            "buckets": buckets,
        }

    @staticmethod
    def _optional_numeric(frame: pd.DataFrame, column: str | None) -> pd.Series | None:
        if not column or column not in frame.columns:
            return None
        return pd.to_numeric(frame[column], errors="coerce")

    @staticmethod
    def _horizon(target: str) -> int | None:
        match = re.search(r"_(\d+)d$", target)
        return int(match.group(1)) if match else None

    @staticmethod
    def _correlation(x: Any, y: Any, method: str) -> float | None:
        left = pd.Series(x, dtype="float64")
        right = pd.Series(y, dtype="float64")
        valid = left.notna() & right.notna()
        if int(valid.sum()) < 2 or left[valid].nunique() < 2 or right[valid].nunique() < 2:
            return None
        value = left[valid].corr(right[valid], method=method)
        return None if pd.isna(value) else float(value)

    @staticmethod
    def _mean(series: pd.Series) -> float | None:
        value = series.mean()
        return None if pd.isna(value) else float(value)

    @staticmethod
    def _median(series: pd.Series) -> float | None:
        value = series.median()
        return None if pd.isna(value) else float(value)
