from __future__ import annotations

import math

from src.quant.feature_schema import QuantFeatureVector, QuantScoreBreakdown


def clip(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)


def clip01(value: float) -> float:
    return clip(value, 0.0, 1.0)


def scale(value: float, low: float, high: float) -> float:
    if high == low:
        return 0.5
    return clip01((value - low) / (high - low))


def scale_optional(value: float | None, low: float, high: float, default: float = 0.5) -> float:
    if value is None or not math.isfinite(value):
        return default
    return scale(value, low, high)


def moderate_volume_score(ratio: float | None) -> float:
    if ratio is None or not math.isfinite(ratio):
        return 50.0
    if ratio < 0.8:
        return 30.0
    if ratio < 1.2:
        return 50.0 + (ratio - 0.8) / 0.4 * 20.0
    if ratio < 2.5:
        return 70.0 + (ratio - 1.2) / 1.3 * 30.0
    if ratio < 4.0:
        return 100.0 - (ratio - 2.5) / 1.5 * 40.0
    return 40.0


class RuleScorer:
    """将量化特征映射为透明、可复现的规则评分。"""

    def score(self, features: QuantFeatureVector) -> QuantScoreBreakdown:
        trend_score = 100.0 * (
            0.35 * scale_optional(features.ma5_ma20_gap, -0.03, 0.05)
            + 0.35 * scale_optional(features.ma20_ma60_gap, -0.05, 0.08)
            + 0.30 * scale_optional(features.close_ma20_gap, -0.05, 0.08)
        )

        momentum_score = 100.0 * (
            0.25 * scale_optional(features.return_3d, -0.03, 0.05)
            + 0.35 * scale_optional(features.return_5d, -0.05, 0.08)
            + 0.40 * scale_optional(features.return_20d, -0.10, 0.20)
        )

        volume_score = (
            0.60 * moderate_volume_score(features.volume_ratio_5d)
            + 0.40 * moderate_volume_score(features.amount_ratio_5d)
        )

        drawdown = None if features.max_drawdown_20d is None else abs(features.max_drawdown_20d)
        risk_score = 100.0 * (
            0.35 * scale_optional(features.volatility_20d, 0.01, 0.05, default=0.4)
            + 0.40 * scale_optional(drawdown, 0.05, 0.25, default=0.4)
            + 0.25 * scale_optional(features.atr_14, 0.015, 0.08, default=0.4)
        )
        if features.is_st is True:
            risk_score = max(risk_score, 85.0)
        if features.is_latest_suspended:
            risk_score = 100.0
        if features.data_quality_score < 50:
            risk_score = max(risk_score, 75.0)

        overheat_score = 100.0 * (
            0.25 * scale_optional(features.return_5d, 0.05, 0.15, default=0.0)
            + 0.25 * scale_optional(features.return_20d, 0.15, 0.40, default=0.0)
            + 0.20 * scale_optional(features.rsi_14, 65.0, 85.0, default=0.0)
            + 0.15 * scale_optional(features.bollinger_position, 0.8, 1.5, default=0.0)
            + 0.15 * scale_optional(features.volume_ratio_5d, 2.5, 5.0, default=0.0)
        )

        if features.macd_hist is None or not math.isfinite(features.macd_hist):
            macd_score = 50.0
        elif features.macd_hist > 0:
            macd_score = 70.0
        else:
            macd_score = 40.0

        support_score = (
            0.35 * trend_score
            + 0.30 * momentum_score
            + 0.20 * volume_score
            + 0.15 * macd_score
        )
        penalty_score = 0.25 * risk_score + 0.20 * overheat_score
        quant_score = clip(support_score - penalty_score + 25.0, 0.0, 100.0)

        probability = 0.35 + 0.40 * quant_score / 100.0
        probability -= 0.10 * risk_score / 100.0
        probability -= 0.08 * overheat_score / 100.0
        probability = clip(probability, 0.05, 0.85)

        if features.is_latest_suspended:
            quant_decision = "reject"
        elif features.data_quality_score < 50:
            quant_decision = "uncertain"
        elif risk_score >= 85:
            quant_decision = "reject"
        elif overheat_score >= 85:
            quant_decision = "reject"
        elif quant_score >= 65 and risk_score < 70 and overheat_score < 75:
            quant_decision = "support"
        elif quant_score <= 45:
            quant_decision = "reject"
        else:
            quant_decision = "uncertain"

        score_tags = self._score_tags(
            trend_score=trend_score,
            momentum_score=momentum_score,
            volume_score=volume_score,
            risk_score=risk_score,
            overheat_score=overheat_score,
            macd_score=macd_score,
        )

        return QuantScoreBreakdown(
            trend_score=clip(trend_score, 0.0, 100.0),
            momentum_score=clip(momentum_score, 0.0, 100.0),
            volume_score=clip(volume_score, 0.0, 100.0),
            risk_score=clip(risk_score, 0.0, 100.0),
            overheat_score=clip(overheat_score, 0.0, 100.0),
            macd_score=macd_score,
            support_score=clip(support_score, 0.0, 100.0),
            penalty_score=clip(penalty_score, 0.0, 100.0),
            quant_score=quant_score,
            heuristic_prob_up_5d=probability,
            quant_decision=quant_decision,
            score_tags=score_tags,
            warnings=list(features.warnings),
            metadata={"rule_version": "rule_v1"},
        )

    @staticmethod
    def _score_tags(
        trend_score: float,
        momentum_score: float,
        volume_score: float,
        risk_score: float,
        overheat_score: float,
        macd_score: float,
    ) -> list[str]:
        tags: list[str] = []
        if trend_score >= 70:
            tags.append("trend_strong")
        elif trend_score <= 40:
            tags.append("trend_weak")
        if momentum_score >= 70:
            tags.append("momentum_strong")
        elif momentum_score <= 40:
            tags.append("momentum_weak")
        if volume_score >= 70:
            tags.append("volume_confirmed")
        if risk_score >= 75:
            tags.append("high_risk")
        if overheat_score >= 75:
            tags.append("overheated")
        if macd_score >= 65:
            tags.append("macd_positive")
        elif macd_score <= 45:
            tags.append("macd_weak")
        return tags
