from __future__ import annotations

from typing import Any

from src.data.realtime_quote_provider import CurrentPriceSnapshot
from src.models.schemas import FinalDecision
from src.recommendation.recommendation_explainer import RecommendationExplainer
from src.recommendation.recommendation_schema import ActionLevel, Recommendation, RecommendationAction


class RecommendationEngine:
    CONTEXT_FIELDS = (
        "latest_close", "quant_score", "trend_score", "momentum_score", "volume_score",
        "risk_score", "overheat_score", "macd_score", "return_5d", "return_20d",
        "ma5_ma20_gap", "ma20_ma60_gap", "close_ma20_gap", "volume_ratio_5d",
        "volume_ratio_20d", "amount_ratio_5d", "amount_ratio_20d", "volatility_20d",
        "max_drawdown_20d", "atr_14", "rsi_14", "macd_hist", "bollinger_position",
    )

    def __init__(self, explainer: RecommendationExplainer | None = None) -> None:
        self.explainer = explainer or RecommendationExplainer()

    def build_from_decision(self, decision: FinalDecision) -> Recommendation:
        context = self.extract_quant_context(decision)
        quant_decision = str(context["quant_decision"])
        risk_score = float(context.get("risk_score") or 0.0)
        overheat_score = float(context.get("overheat_score") or 0.0)
        final_level = self._enum_value(decision.final_level)
        source_type = self.extract_source_type(decision)

        if source_type == "manual_watch":
            action, level = self._manual_watch_action(
                quant_decision, risk_score, overheat_score, decision.final_score, final_level
            )
        elif (
            decision.ai_view == "positive" and quant_decision == "support"
            and risk_score < 70 and overheat_score < 75 and decision.final_score >= 60
        ):
            action = RecommendationAction.BUY_CANDIDATE
            level = ActionLevel.HIGH if decision.final_score >= 75 and risk_score < 50 and overheat_score < 60 else ActionLevel.MEDIUM
        elif decision.ai_view == "positive" and quant_decision == "reject":
            action = RecommendationAction.RISK_WARNING
            level = ActionLevel.HIGH if risk_score >= 90 or overheat_score >= 90 else ActionLevel.MEDIUM
        elif quant_decision == "reject" or risk_score >= 85 or overheat_score >= 85 or final_level == "avoid" or decision.ai_view == "negative":
            action = RecommendationAction.AVOID
            level = ActionLevel.HIGH if risk_score >= 90 or overheat_score >= 90 else ActionLevel.MEDIUM
        elif ((decision.ai_view == "positive" and quant_decision == "uncertain")
              or (decision.ai_view == "neutral" and quant_decision == "support")
              or final_level == "watch"):
            action, level = RecommendationAction.WATCH, ActionLevel.MEDIUM
        else:
            action, level = RecommendationAction.WATCH, ActionLevel.LOW

        recommendation = Recommendation(
            symbol=decision.symbol, stock_name=decision.stock_name,
            source_type=source_type, as_of_time=decision.as_of_time,
            action=action, action_level=level,
            confidence=max(0.2, min(1.0, decision.final_score / 100.0)),
            suggested_horizon_days=self._suggested_horizons(decision),
            ai_view=decision.ai_view, quant_decision=quant_decision,
            final_level=final_level, final_score=decision.final_score,
            metadata={**context, "source_type": source_type},
        )
        return self.explainer.explain(decision, recommendation)

    def build_many(self, decisions: list[FinalDecision]) -> list[Recommendation]:
        return sorted(
            (self.build_from_decision(item) for item in decisions),
            key=lambda item: item.final_score or 0.0, reverse=True,
        )

    @staticmethod
    def apply_intraday_overlay(
        recommendation: Recommendation,
        price: CurrentPriceSnapshot,
    ) -> Recommendation:
        """Attach a quote snapshot and protect buy candidates without changing daily indicators."""
        original_action = recommendation.action
        action = original_action
        level = recommendation.action_level
        intraday_reason = ""
        intraday_confirmed = bool(
            price.is_realtime and original_action == RecommendationAction.BUY_CANDIDATE
        )
        pct_change = price.realtime_pct_change
        quote = price.quote
        if pct_change is None and quote is not None and quote.prev_close and price.current_price:
            pct_change = price.current_price / quote.prev_close - 1.0

        if original_action == RecommendationAction.BUY_CANDIDATE:
            if not price.is_realtime:
                action, level = RecommendationAction.WATCH, ActionLevel.MEDIUM
                intraday_confirmed = False
                intraday_reason = "当前仅有日线数据，未获得盘中实时确认，不直接作为买入候选。"
            elif pct_change is not None and pct_change < -0.02:
                action, level = RecommendationAction.RISK_WARNING, ActionLevel.HIGH
                intraday_confirmed = False
                intraday_reason = "盘中跌幅较大，优先风险控制。"
            elif pct_change is not None and pct_change < 0:
                action, level = RecommendationAction.WATCH, ActionLevel.MEDIUM
                intraday_confirmed = False
                intraday_reason = "盘中价格弱于昨收，买入信号未确认。"
            else:
                latest_daily = RecommendationEngine._number(
                    recommendation.metadata.get("latest_close")
                )
                close_ma20_gap = RecommendationEngine._number(
                    recommendation.metadata.get("close_ma20_gap")
                )
                realtime = RecommendationEngine._number(price.current_price)
                ma20 = None
                if (
                    latest_daily is not None
                    and close_ma20_gap is not None
                    and 1.0 + close_ma20_gap != 0
                ):
                    ma20 = latest_daily / (1.0 + close_ma20_gap)
                if realtime is not None and ma20 is not None and realtime < ma20:
                    action = RecommendationAction.RISK_WARNING if (pct_change or 0.0) <= -0.01 else RecommendationAction.WATCH
                    level = ActionLevel.HIGH if action == RecommendationAction.RISK_WARNING else ActionLevel.MEDIUM
                    intraday_confirmed = False
                    intraday_reason = "实时价格已跌破 20 日均线附近，日线买入条件未获得盘中确认。"
                else:
                    overheat = RecommendationEngine._number(
                        recommendation.metadata.get("overheat_score")
                    ) or 0.0
                    pullback = None
                    if quote is not None and quote.high and realtime:
                        pullback = realtime / quote.high - 1.0
                    if overheat >= 75 and pullback is not None and pullback <= -0.02:
                        action, level = RecommendationAction.RISK_WARNING, ActionLevel.HIGH
                        intraday_confirmed = False
                        intraday_reason = "短期过热且盘中冲高回落，优先观察回落风险。"
                    else:
                        intraday_reason = "已获得盘中实时行情，当前价格未触发买入候选降级条件。"
        elif price.is_realtime:
            intraday_reason = "已叠加盘中实时行情；原始动作不属于买入候选，无需升级动作。"
        else:
            intraday_reason = "当前仅有最新完整日线，未获得盘中实时行情。"

        metadata = {
            **recommendation.metadata,
            "original_action": original_action.value,
            "original_final_score": recommendation.final_score,
            "intraday_confirmed": intraday_confirmed,
            "intraday_reason": intraday_reason,
            "realtime_pct_change": pct_change,
            "realtime_price": price.current_price if price.is_realtime else None,
            "realtime_source": price.realtime_source,
            "current_price": price.current_price,
            "price_time": price.price_time.isoformat() if price.price_time else None,
            "price_source": price.price_source,
            "is_realtime": price.is_realtime,
            "is_stale": price.is_stale,
            "price_warning": price.warning,
        }
        reasons = list(recommendation.reason)
        risks = list(recommendation.risks)
        if intraday_reason and intraday_reason not in reasons:
            reasons.insert(0, intraday_reason)
        if price.warning and price.warning not in risks:
            risks.append(price.warning)
        return recommendation.model_copy(update={
            "action": action,
            "action_level": level,
            "reason": reasons,
            "risks": risks,
            "metadata": metadata,
        })

    @staticmethod
    def _number(value: Any) -> float | None:
        try:
            return None if value is None else float(value)
        except (TypeError, ValueError, OverflowError):
            return None

    @staticmethod
    def _manual_watch_action(quant_decision: str, risk_score: float, overheat_score: float,
                             final_score: float, final_level: str) -> tuple[RecommendationAction, ActionLevel]:
        if quant_decision == "reject":
            if final_level == "avoid" or risk_score >= 85 or overheat_score >= 85:
                return RecommendationAction.AVOID, ActionLevel.HIGH
            return RecommendationAction.RISK_WARNING, ActionLevel.MEDIUM
        if quant_decision == "support":
            return RecommendationAction.WATCH, ActionLevel.HIGH if final_score >= 70 else ActionLevel.MEDIUM
        return RecommendationAction.WATCH, ActionLevel.MEDIUM if final_score >= 50 else ActionLevel.LOW

    @classmethod
    def extract_quant_context(cls, decision: FinalDecision) -> dict[str, Any]:
        quant = decision.quant_result
        features = quant.features
        extra = features.extra if features is not None and isinstance(features.extra, dict) else {}
        vector = extra.get("feature_vector") if isinstance(extra.get("feature_vector"), dict) else {}
        scores = extra.get("score_breakdown") if isinstance(extra.get("score_breakdown"), dict) else {}
        quant_decision = extra.get("internal_quant_decision") or scores.get("quant_decision")
        if quant_decision not in {"support", "uncertain", "reject"}:
            quant_decision = decision.quant_view
        if quant_decision == "neutral":
            quant_decision = "uncertain"

        fallbacks = {
            "risk_score": quant.risk_score, "overheat_score": quant.overheat_score,
            "trend_score": quant.trend_score, "momentum_score": quant.momentum_score,
        }

        def number(name: str) -> float | None:
            value = scores.get(name)
            if value is None:
                value = vector.get(name)
            if value is None and features is not None:
                value = getattr(features, name, None)
            if value is None:
                value = extra.get(name)
            if value is None:
                value = fallbacks.get(name)
            try:
                return None if value is None else float(value)
            except (TypeError, ValueError, OverflowError):
                return None

        context = {name: number(name) for name in cls.CONTEXT_FIELDS}
        context.update({
            "quant_decision": quant_decision,
            "risk_score": context.get("risk_score") or 0.0,
            "overheat_score": context.get("overheat_score") or 0.0,
            "feature_vector": vector,
            "score_breakdown": scores,
        })
        return context

    @staticmethod
    def extract_source_type(decision: FinalDecision) -> str:
        metadata = decision.ai_candidate.metadata or {}
        nested = metadata.get("candidate_metadata") if isinstance(metadata.get("candidate_metadata"), dict) else {}
        signal = metadata.get("signal_candidate") if isinstance(metadata.get("signal_candidate"), dict) else {}
        signal_meta = signal.get("metadata") if isinstance(signal.get("metadata"), dict) else {}
        value = metadata.get("source_type") or nested.get("source_type") or signal_meta.get("source_type")
        if value == "manual_watch":
            return "manual_watch"
        return "ai_signal" if value in (None, "", "ai_signal") else "unknown"

    @staticmethod
    def _suggested_horizons(decision: FinalDecision) -> list[int]:
        raw = (decision.ai_candidate.metadata or {}).get("expected_horizon_days") or [3, 5]
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
