from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel, Field


class MLGuardResult(BaseModel):
    raw_score: float = Field(ge=0, le=1)
    guarded_score: float = Field(ge=0, le=1)
    guard_flags: list[str] = Field(default_factory=list)
    guard_action: Literal["allow", "caution", "downrank", "block"]
    guard_reason: str
    penalty: float = Field(ge=0)


class MLScoringGuard:
    """Research-only safety overlay for cross-sectional ML ranking scores."""

    def apply(self, row: Mapping[str, Any] | Any, raw_score: float) -> MLGuardResult:
        raw = self._clip(self._number(raw_score) or 0.0, 0.0, 1.0)
        flags: list[str] = []
        penalty = 0.0

        breadth = self._value(row, "breadth_up_ratio_5d")
        if breadth is not None and breadth < 0.40:
            flags.append("weak_breadth")
            penalty += 0.15

        ma_breadth = self._value(row, "breadth_above_ma20_ratio")
        if ma_breadth is not None and ma_breadth < 0.40:
            flags.append("weak_ma_breadth")
            penalty += 0.08

        return_20d = self._value(row, "return_20d")
        if return_20d is not None and return_20d > 0.15:
            flags.append("chase_20d")
            penalty += 0.10
        if return_20d is not None and return_20d > 0.25:
            flags.append("extreme_chase_20d")
            penalty += 0.20

        return_5d = self._value(row, "return_5d")
        if return_5d is not None and return_5d > 0.08:
            flags.append("chase_5d")
            penalty += 0.08
        if return_5d is not None and return_5d > 0.12:
            flags.append("extreme_chase_5d")
            penalty += 0.15

        overheat = self._value(row, "overheat_score")
        if overheat is not None and overheat > 50:
            flags.append("high_overheat")
            penalty += 0.10

        risk = self._value(row, "risk_score")
        if risk is not None and risk > 55:
            flags.append("high_risk_score")
            penalty += 0.10

        amount_ratio = self._value(row, "amount_ratio_5d")
        if (
            amount_ratio is not None and amount_ratio > 3
            and return_5d is not None and return_5d > 0.08
        ):
            flags.append("volume_chase")
            penalty += 0.10

        relative_strength = self._value(row, "relative_to_hs300_return_5d")
        if (
            relative_strength is not None and relative_strength > 0.05
            and return_20d is not None and return_20d > 0.15
        ):
            flags.append("relative_strength_chase")
            penalty += 0.08

        guarded = self._clip(raw - penalty, 0.0, 1.0)
        if penalty >= 0.25:
            action = "block"
        elif (
            penalty >= 0.12
            or "weak_breadth" in flags
            or "extreme_chase_20d" in flags
            or "extreme_chase_5d" in flags
        ):
            action = "downrank"
        elif penalty > 0:
            action = "caution"
        else:
            action = "allow"
        reason = (
            "No scoring guard triggered."
            if not flags
            else f"Applied {action} guard: {', '.join(flags)}; penalty={penalty:.2f}."
        )
        return MLGuardResult(
            raw_score=raw,
            guarded_score=guarded,
            guard_flags=flags,
            guard_action=action,
            guard_reason=reason,
            penalty=round(penalty, 10),
        )

    @staticmethod
    def _value(row: Mapping[str, Any] | Any, key: str) -> float | None:
        if isinstance(row, Mapping):
            value = row.get(key)
        elif hasattr(row, "get"):
            value = row.get(key)
        else:
            value = getattr(row, key, None)
        return MLScoringGuard._number(value)

    @staticmethod
    def _number(value: Any) -> float | None:
        try:
            result = float(value)
        except (TypeError, ValueError, OverflowError):
            return None
        return result if math.isfinite(result) else None

    @staticmethod
    def _clip(value: float, low: float, high: float) -> float:
        return min(max(value, low), high)
