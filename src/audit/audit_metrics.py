from __future__ import annotations

import math
from statistics import median
from typing import Any

import pandas as pd

from src.audit.audit_schema import AuditSample


class AuditMetricsBuilder:
    ACTIONS = ("buy_candidate", "watch", "risk_warning", "avoid")
    QUANT_DECISIONS = ("support", "uncertain", "reject")

    @classmethod
    def build_summary(cls, samples: list[AuditSample]) -> dict[str, Any]:
        action_metrics = cls._group_metrics(samples, "action", cls.ACTIONS)
        quant_metrics = cls._group_metrics(samples, "quant_decision", cls.QUANT_DECISIONS)
        action_distribution = {
            action: sum(item.action == action for item in samples) for action in cls.ACTIONS
        }
        for item in samples:
            if item.action and item.action not in action_distribution:
                action_distribution[item.action] = action_distribution.get(item.action, 0) + 1

        support = quant_metrics["support"]
        reject = quant_metrics["reject"]
        buy = action_metrics["buy_candidate"]
        risk = action_metrics["risk_warning"]
        ranking_warning = (
            cls._lower_hit_rate(buy, risk)
            or cls._lower_hit_rate(support, reject)
        )
        return {
            "samples_count": len(samples),
            "complete_samples": sum(item.is_complete for item in samples),
            "error_samples": sum(item.error_message is not None for item in samples),
            "action_distribution": action_distribution,
            "avg_future_return_5d": cls._mean(item.future_return_5d for item in samples),
            "avg_future_return_10d": cls._mean(item.future_return_10d for item in samples),
            "median_future_return_5d": cls._median(item.future_return_5d for item in samples),
            "median_future_return_10d": cls._median(item.future_return_10d for item in samples),
            "hit_rate_5d": cls._hit_rate(item.future_return_5d for item in samples),
            "hit_rate_10d": cls._hit_rate(item.future_return_10d for item in samples),
            "avg_max_drawdown_5d": cls._mean(item.future_max_drawdown_5d for item in samples),
            "avg_max_drawdown_10d": cls._mean(item.future_max_drawdown_10d for item in samples),
            "action_metrics": action_metrics,
            "quant_decision_metrics": quant_metrics,
            "score_future_return_corr_5d": cls._correlation(samples, "quant_score", "future_return_5d"),
            "score_future_return_corr_10d": cls._correlation(samples, "quant_score", "future_return_10d"),
            "final_score_future_return_corr_5d": cls._correlation(samples, "final_score", "future_return_5d"),
            "final_score_future_return_corr_10d": cls._correlation(samples, "final_score", "future_return_10d"),
            "ranking_warning": ranking_warning,
            "sample_note": "独立 quant_only_audit 结果，仅用于算法审查，不代表正式推荐或真实交易结论。",
        }

    @classmethod
    def build_cases(cls, samples: list[AuditSample]) -> list[dict[str, Any]]:
        cases: list[dict[str, Any]] = []
        for item in samples:
            case_type: str | None = None
            if item.action == "buy_candidate" and cls._le(item.future_return_5d, 0.0):
                case_type = "bad_buy_candidate"
            elif item.action in {"watch", "risk_warning", "avoid"} and cls._ge(item.future_return_5d, 0.05):
                case_type = "missed_opportunity"
            elif item.action == "risk_warning" and cls._le(item.future_max_drawdown_5d, -0.03):
                case_type = "good_risk_warning"
            elif (
                item.action == "risk_warning"
                and cls._ge(item.future_return_5d, 0.05)
                and item.future_max_drawdown_5d is not None
                and item.future_max_drawdown_5d > -0.03
            ):
                case_type = "bad_risk_warning"
            if case_type:
                row = item.model_dump(mode="json")
                row["case_type"] = case_type
                cases.append(row)
        return cases

    @classmethod
    def _group_metrics(
        cls,
        samples: list[AuditSample],
        field: str,
        expected_groups: tuple[str, ...],
    ) -> dict[str, dict[str, Any]]:
        groups = list(expected_groups)
        groups.extend(
            value for value in dict.fromkeys(getattr(item, field) for item in samples)
            if value and value not in groups
        )
        result: dict[str, dict[str, Any]] = {}
        for group in groups:
            selected = [item for item in samples if getattr(item, field) == group]
            result[group] = {
                "count": len(selected),
                "hit_rate_5d": cls._hit_rate(item.future_return_5d for item in selected),
                "hit_rate_10d": cls._hit_rate(item.future_return_10d for item in selected),
                "avg_return_5d": cls._mean(item.future_return_5d for item in selected),
                "avg_return_10d": cls._mean(item.future_return_10d for item in selected),
                "median_return_5d": cls._median(item.future_return_5d for item in selected),
                "median_return_10d": cls._median(item.future_return_10d for item in selected),
                "avg_max_drawdown_5d": cls._mean(item.future_max_drawdown_5d for item in selected),
                "avg_max_drawdown_10d": cls._mean(item.future_max_drawdown_10d for item in selected),
            }
        return result

    @staticmethod
    def _valid(values) -> list[float]:
        result: list[float] = []
        for value in values:
            if value is None:
                continue
            number = float(value)
            if math.isfinite(number):
                result.append(number)
        return result

    @classmethod
    def _mean(cls, values) -> float | None:
        valid = cls._valid(values)
        return sum(valid) / len(valid) if valid else None

    @classmethod
    def _median(cls, values) -> float | None:
        valid = cls._valid(values)
        return median(valid) if valid else None

    @classmethod
    def _hit_rate(cls, values) -> float | None:
        valid = cls._valid(values)
        return sum(value > 0 for value in valid) / len(valid) if valid else None

    @staticmethod
    def _correlation(samples: list[AuditSample], left: str, right: str) -> float | None:
        frame = pd.DataFrame([
            {"left": getattr(item, left), "right": getattr(item, right)} for item in samples
        ]).dropna()
        if len(frame) < 2 or frame["left"].nunique() < 2 or frame["right"].nunique() < 2:
            return None
        value = frame["left"].corr(frame["right"], method="pearson")
        return None if pd.isna(value) else float(value)

    @staticmethod
    def _lower_hit_rate(left: dict[str, Any], right: dict[str, Any]) -> bool:
        return (
            left.get("count", 0) > 0
            and right.get("count", 0) > 0
            and left.get("hit_rate_5d") is not None
            and right.get("hit_rate_5d") is not None
            and left["hit_rate_5d"] < right["hit_rate_5d"]
        )

    @staticmethod
    def _le(value: float | None, threshold: float) -> bool:
        return value is not None and value <= threshold

    @staticmethod
    def _ge(value: float | None, threshold: float) -> bool:
        return value is not None and value >= threshold
