from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


class RuleRevisionLab:
    VERSION_NAMES = (
        "current_reconstructed",
        "anti_chase_v1",
        "mean_reversion_guard_v1",
        "reverse_test_only",
    )
    SCORE_FEATURE_COLUMNS = [
        "quant_score", "quant_decision", "return_5d", "return_20d", "rsi_14",
        "bollinger_position", "volume_ratio_5d", "trend_score", "momentum_score",
        "volume_score", "risk_score", "overheat_score", "macd_score",
        "max_drawdown_20d",
    ]
    SAMPLE_NOTE = "当前结果来自测试型 historical batch，不代表真实历史回测表现。"

    def __init__(self, dataset_path: str = "data/ml_dataset.csv"):
        self.dataset_path = Path(dataset_path)

    def load_complete(self) -> pd.DataFrame:
        if not self.dataset_path.exists():
            raise FileNotFoundError(str(self.dataset_path))
        frame = pd.read_csv(self.dataset_path)
        if "feedback_status" not in frame:
            return frame.iloc[0:0].copy()
        status = frame["feedback_status"].astype("string").str.strip().str.lower()
        complete = frame.loc[status == "complete"].copy().reset_index(drop=True)
        for column in set(self.SCORE_FEATURE_COLUMNS + ["future_return_5d"]):
            if column in complete and column != "quant_decision":
                complete[column] = pd.to_numeric(complete[column], errors="coerce")
        complete["_hit_5d_numeric"] = self._binary_target(
            complete.get("hit_5d", pd.Series(index=complete.index, dtype="object"))
        )
        return complete

    def evaluate_all(self) -> dict[str, dict[str, Any]]:
        complete = self.load_complete()
        versions = self._score_all(complete)
        return {
            name: self._evaluate_version(complete, name, score, decision)
            for name, (score, decision) in versions.items()
        }

    def print_report(self) -> None:
        complete = self.load_complete()
        print("StockLens Rule Revision Lab v1.0\n")
        print(f"complete samples: {len(complete)}")
        if complete.empty:
            print("没有 complete 样本，暂时无法运行离线规则实验。")
            print(f"\n{self.SAMPLE_NOTE}")
            return

        results = self.evaluate_all()
        for name in self.VERSION_NAMES:
            metrics = results[name]
            print(f"\n{name}:")
            print(
                f"  support: count={metrics['support_count']}, "
                f"hit_rate={self._percent(metrics['support_hit_rate_5d'])}, "
                f"avg_return={self._percent(metrics['support_avg_return_5d'])}, "
                f"bad_support={metrics['bad_support_count']}"
            )
            print(
                f"  uncertain: count={metrics['uncertain_count']}, "
                f"hit_rate={self._percent(metrics['uncertain_hit_rate_5d'])}, "
                f"avg_return={self._percent(metrics['uncertain_avg_return_5d'])}"
            )
            print(
                f"  reject: count={metrics['reject_count']}, "
                f"hit_rate={self._percent(metrics['reject_hit_rate_5d'])}, "
                f"avg_return={self._percent(metrics['reject_avg_return_5d'])}, "
                f"missed_reject={metrics['missed_reject_count']}"
            )
            print(
                f"  score_return_corr={self._number(metrics['score_future_return_corr'])}, "
                f"score_hit_auc={self._number(metrics['score_hit_auc'])}, "
                f"ranking_warning={metrics['ranking_warning']}"
            )

        print("\nreverse_test_only 仅用于验证当前样本中的反向关系，不能作为正式策略。")
        print(self.SAMPLE_NOTE)

    def export_results(self, output_dir: str = "data/rule_experiments") -> None:
        complete = self.load_complete()
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        versions = self._score_all(complete)

        summary = [
            self._evaluate_version(complete, name, score, decision)
            for name, (score, decision) in versions.items()
        ]
        pd.DataFrame(summary).to_csv(
            output_path / "rule_revision_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )

        base_columns = [
            "symbol", "as_of_time", "future_return_5d", "hit_5d",
            "quant_score", "quant_decision",
        ]
        cases = complete.copy()
        for column in base_columns:
            if column not in cases:
                cases[column] = pd.NA
        cases = cases[base_columns].copy()
        for name, (score, decision) in versions.items():
            cases[f"{name}_score"] = score.to_numpy()
            cases[f"{name}_decision"] = decision.to_numpy()
        cases.to_csv(
            output_path / "rule_revision_cases.csv",
            index=False,
            encoding="utf-8-sig",
        )

    def _score_all(self, frame: pd.DataFrame) -> dict[str, tuple[pd.Series, pd.Series]]:
        # 本方法只读取特征与原规则输出；未来收益和 hit 标签不会参与任何评分公式。
        current_score = self._feature(frame, "quant_score", 50.0).clip(0.0, 100.0)
        current_fallback = self._decision_from_score(current_score)
        if "quant_decision" in frame:
            current_decision = frame["quant_decision"].astype("string").str.strip().str.lower()
            current_decision = current_decision.where(
                current_decision.isin(["support", "uncertain", "reject"]),
                current_fallback,
            )
        else:
            current_decision = current_fallback

        trend = self._feature(frame, "trend_score", 50.0)
        momentum = self._feature(frame, "momentum_score", 50.0)
        volume = self._feature(frame, "volume_score", 50.0)
        macd = self._feature(frame, "macd_score", 50.0)
        risk = self._feature(frame, "risk_score", 50.0)
        overheat = self._feature(frame, "overheat_score", 50.0)
        return_5d = self._feature(frame, "return_5d", 0.0)
        return_20d = self._feature(frame, "return_20d", 0.0)
        rsi = self._feature(frame, "rsi_14", 50.0)
        bollinger = self._feature(frame, "bollinger_position", 0.0)
        volume_ratio = self._feature(frame, "volume_ratio_5d", 1.0)
        drawdown = self._feature(frame, "max_drawdown_20d", 0.0)

        anti_chase = (
            0.25 * trend
            + 0.20 * momentum
            + 0.15 * volume
            + 0.15 * macd
            + 25.0
            - 0.30 * overheat
            - 0.20 * risk
        )
        anti_extra = pd.Series(0.0, index=frame.index)
        anti_extra += (return_5d > 0.05).astype(float) * 8.0
        anti_extra += (return_20d > 0.15).astype(float) * 10.0
        anti_extra += (rsi > 70.0).astype(float) * 8.0
        anti_extra += (bollinger > 1.0).astype(float) * 6.0
        anti_extra += (volume_ratio > 2.5).astype(float) * 6.0
        anti_chase = (anti_chase - anti_extra).clip(0.0, 100.0)

        mean_reversion = (
            0.30 * trend
            + 0.20 * macd
            + 0.15 * volume
            + 0.10 * momentum
            + 25.0
        )
        guard_bonus = return_5d.between(-0.03, 0.05, inclusive="both") & (trend > 50.0)
        mean_reversion += guard_bonus.astype(float) * 8.0
        mean_reversion -= (return_20d > 0.20).astype(float) * 12.0
        mean_reversion -= (overheat > 70.0).astype(float) * 15.0
        mean_reversion -= (risk > 75.0).astype(float) * 10.0
        mean_reversion -= (drawdown < -0.15).astype(float) * 8.0
        mean_reversion = mean_reversion.clip(0.0, 100.0)

        reverse = (100.0 - current_score).clip(0.0, 100.0)
        return {
            "current_reconstructed": (current_score, current_decision),
            "anti_chase_v1": (anti_chase, self._decision_from_score(anti_chase)),
            "mean_reversion_guard_v1": (
                mean_reversion,
                self._decision_from_score(mean_reversion),
            ),
            "reverse_test_only": (reverse, self._decision_from_score(reverse)),
        }

    def _evaluate_version(
        self,
        frame: pd.DataFrame,
        version_name: str,
        score: pd.Series,
        decision: pd.Series,
    ) -> dict[str, Any]:
        future_return = self._feature(frame, "future_return_5d", float("nan"))
        hits = frame.get(
            "_hit_5d_numeric",
            pd.Series(float("nan"), index=frame.index, dtype="float64"),
        )
        metrics: dict[str, Any] = {"version_name": version_name}
        group_values: dict[str, dict[str, Any]] = {}
        for group in ("support", "uncertain", "reject"):
            mask = decision == group
            group_values[group] = {
                "count": int(mask.sum()),
                "hit_rate": self._mean_or_none(hits.loc[mask]),
                "avg_return": self._mean_or_none(future_return.loc[mask]),
            }
            metrics[f"{group}_count"] = group_values[group]["count"]
            metrics[f"{group}_hit_rate_5d"] = group_values[group]["hit_rate"]
            metrics[f"{group}_avg_return_5d"] = group_values[group]["avg_return"]

        metrics["bad_support_count"] = int(
            ((decision == "support") & future_return.notna() & (future_return <= 0)).sum()
        )
        metrics["missed_reject_count"] = int(
            ((decision == "reject") & future_return.notna() & (future_return > 0)).sum()
        )
        metrics["score_future_return_corr"] = self._correlation(score, future_return)
        metrics["score_hit_auc"] = self._auc(score, hits)
        support_rate = group_values["support"]["hit_rate"]
        reject_rate = group_values["reject"]["hit_rate"]
        metrics["ranking_warning"] = bool(
            support_rate is not None
            and reject_rate is not None
            and support_rate < reject_rate
        )
        metrics["sample_note"] = self.SAMPLE_NOTE
        return metrics

    @staticmethod
    def _decision_from_score(score: pd.Series) -> pd.Series:
        result = pd.Series("reject", index=score.index, dtype="string")
        result.loc[(score >= 45.0) & (score < 65.0)] = "uncertain"
        result.loc[score >= 65.0] = "support"
        return result

    @staticmethod
    def _feature(frame: pd.DataFrame, column: str, default: float) -> pd.Series:
        if column not in frame:
            return pd.Series(default, index=frame.index, dtype="float64")
        return pd.to_numeric(frame[column], errors="coerce").fillna(default).astype("float64")

    @staticmethod
    def _binary_target(series: pd.Series) -> pd.Series:
        mapping = {
            "true": 1.0, "1": 1.0, "1.0": 1.0, "yes": 1.0,
            "false": 0.0, "0": 0.0, "0.0": 0.0, "no": 0.0,
        }
        return series.astype("string").str.strip().str.lower().map(mapping).astype("float64")

    @staticmethod
    def _mean_or_none(series: pd.Series) -> float | None:
        clean = pd.to_numeric(series, errors="coerce").dropna()
        return None if clean.empty else float(clean.mean())

    @staticmethod
    def _correlation(score: pd.Series, future_return: pd.Series) -> float | None:
        paired = pd.DataFrame({"score": score, "future_return": future_return}).dropna()
        if len(paired) < 3 or paired["score"].nunique() < 2 or paired["future_return"].nunique() < 2:
            return None
        return float(paired["score"].corr(paired["future_return"], method="pearson"))

    @staticmethod
    def _auc(score: pd.Series, hits: pd.Series) -> float | None:
        paired = pd.DataFrame({"score": score, "hit": hits}).dropna()
        if paired.empty or paired["hit"].nunique() < 2:
            return None
        try:
            from sklearn.metrics import roc_auc_score
        except ImportError:
            return None
        try:
            return float(roc_auc_score(paired["hit"], paired["score"]))
        except ValueError:
            return None

    @staticmethod
    def _percent(value: float | None) -> str:
        return "-" if value is None or pd.isna(value) else f"{value:.2%}"

    @staticmethod
    def _number(value: float | None) -> str:
        return "-" if value is None or pd.isna(value) else f"{value:.6f}"
