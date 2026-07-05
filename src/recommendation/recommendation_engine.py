from __future__ import annotations

from typing import Any

from src.models.schemas import FinalDecision
from src.recommendation.recommendation_explainer import RecommendationExplainer
from src.recommendation.recommendation_schema import (
    ActionLevel,
    Recommendation,
    RecommendationAction,
)


class RecommendationEngine:
    def __init__(self, explainer: RecommendationExplainer | None = None) -> None:
        self.explainer = explainer or RecommendationExplainer()

    def build_from_decision(self, decision: FinalDecision) -> Recommendation:
        context = self.extract_quant_context(decision)
        quant_decision = context["quant_decision"]
        risk_score = context["risk_score"]
        overheat_score = context["overheat_score"]
        final_level = self._enum_value(decision.final_level)

        if (
            decision.ai_view == "positive"
            and quant_decision == "support"
            and risk_score < 70
            and overheat_score < 75
            and decision.final_score >= 60
        ):
            action = RecommendationAction.BUY_CANDIDATE
            level = (
                ActionLevel.HIGH
                if decision.final_score >= 75 and risk_score < 50 and overheat_score < 60
                else ActionLevel.MEDIUM
            )
        elif decision.ai_view == "positive" and quant_decision == "reject":
            action = RecommendationAction.RISK_WARNING
            level = ActionLevel.HIGH if risk_score >= 90 or overheat_score >= 90 else ActionLevel.MEDIUM
        elif (
            quant_decision == "reject"
            or risk_score >= 85
            or overheat_score >= 85
            or final_level == "avoid"
            or decision.ai_view == "negative"
        ):
            action = RecommendationAction.AVOID
            level = ActionLevel.HIGH if risk_score >= 90 or overheat_score >= 90 else ActionLevel.MEDIUM
        elif (
            (decision.ai_view == "positive" and quant_decision == "uncertain")
            or (decision.ai_view == "neutral" and quant_decision == "support")
            or final_level == "watch"
        ):
            action = RecommendationAction.WATCH
            level = ActionLevel.MEDIUM
        else:
            action = RecommendationAction.WATCH
            level = ActionLevel.LOW

        horizons = self._suggested_horizons(decision)
        recommendation = Recommendation(
            symbol=decision.symbol,
            stock_name=decision.stock_name,
            as_of_time=decision.as_of_time,
            action=action,
            action_level=level,
            confidence=max(0.2, min(1.0, decision.final_score / 100.0)),
            suggested_horizon_days=horizons,
            ai_view=decision.ai_view,
            quant_decision=quant_decision,
            final_level=final_level,
            final_score=decision.final_score,
            metadata=context,
        )
        return self.explainer.explain(decision, recommendation)

    def build_many(self, decisions: list[FinalDecision]) -> list[Recommendation]:
        recommendations = [self.build_from_decision(decision) for decision in decisions]
        return sorted(recommendations, key=lambda item: item.final_score or 0.0, reverse=True)

    @staticmethod
    def extract_quant_context(decision: FinalDecision) -> dict[str, Any]:
        quant = decision.quant_result
        features = quant.features
        extra = features.extra if features is not None and isinstance(features.extra, dict) else {}
        feature_vector = extra.get("feature_vector") if isinstance(extra.get("feature_vector"), dict) else {}
        score_breakdown = extra.get("score_breakdown") if isinstance(extra.get("score_breakdown"), dict) else {}

        quant_decision = extra.get("internal_quant_decision") or score_breakdown.get("quant_decision")
        if quant_decision not in {"support", "uncertain", "reject"}:
            quant_decision = decision.quant_view
        if quant_decision == "neutral":
            quant_decision = "uncertain"

        def number(name: str, fallback: float | None = None) -> float | None:
            value = score_breakdown.get(name)
            if value is None:
                value = feature_vector.get(name)
            if value is None and features is not None:
                value = getattr(features, name, None)
            if value is None:
                value = fallback
            try:
                return None if value is None else float(value)
            except (TypeError, ValueError):
                return fallback

        return {
            "quant_decision": quant_decision,
            "risk_score": number("risk_score", quant.risk_score) or 0.0,
            "overheat_score": number("overheat_score", quant.overheat_score) or 0.0,
            "trend_score": number("trend_score", quant.trend_score),
            "momentum_score": number("momentum_score", quant.momentum_score),
            "volume_score": number("volume_score"),
            "macd_score": number("macd_score"),
            "quant_score": number("quant_score"),
            "latest_close": number("latest_close"),
            "rsi_14": number("rsi_14"),
            "close_ma20_gap": number("close_ma20_gap"),
            "macd_hist": number("macd_hist"),
            "volume_ratio_5d": number("volume_ratio_5d"),
            "feature_vector": feature_vector,
            "score_breakdown": score_breakdown,
        }

    @staticmethod
    def _suggested_horizons(decision: FinalDecision) -> list[int]:
        metadata = decision.ai_candidate.metadata or {}
        raw = metadata.get("expected_horizon_days") or [3, 5]
        result: list[int] = []
        if isinstance(raw, list):
            for value in raw:
                try:
                    days = int(value)
                except (TypeError, ValueError):
                    continue
                if days > 0 and days not in result:
                    result.append(days)
        return result or [3, 5]

    @staticmethod
    def _enum_value(value: Any) -> str:
        return str(getattr(value, "value", value))
