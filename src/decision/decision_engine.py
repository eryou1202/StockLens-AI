from __future__ import annotations

from src.models.schemas import AICandidate, DecisionLevel, FinalDecision, QuantResult


class DecisionEngine:
    """
    AI 信息 + 量化曲线融合判断层。
    """

    def merge(self, ai_signal: AICandidate, quant_signal: QuantResult) -> FinalDecision:
        ai_score = self._calc_ai_score(ai_signal)
        quant_score = self._calc_quant_score(quant_signal)
        final_score = max(0, min(100, ai_score * 0.45 + quant_score * 0.55))

        ai_view = self._ai_view(ai_signal)
        quant_view = self._quant_view(quant_signal)

        if final_score >= 75 and quant_view == "support":
            level = DecisionLevel.STRONG_WATCH
        elif final_score >= 55:
            level = DecisionLevel.WATCH
        elif final_score >= 40:
            level = DecisionLevel.RISKY
        else:
            level = DecisionLevel.AVOID

        reasons = [
            f"AI事件类型：{ai_signal.event_type}",
            f"AI摘要：{ai_signal.event_summary}",
            f"量化趋势：{quant_signal.trend_state}",
            f"量化风险分：{quant_signal.risk_score:.1f}",
        ]
        if ai_signal.risk_flags:
            reasons.append("AI风险提示：" + "；".join(ai_signal.risk_flags))
        reasons.extend(quant_signal.quant_reason)

        return FinalDecision(
            symbol=ai_signal.stock_code,
            stock_name=ai_signal.stock_name,
            as_of_time=ai_signal.as_of_time,
            ai_view=ai_view,
            quant_view=quant_view,
            final_level=level,
            final_score=final_score,
            ai_summary=ai_signal.event_summary,
            quant_summary="；".join(quant_signal.quant_reason),
            final_reason=reasons,
            ai_candidate=ai_signal,
            quant_result=quant_signal,
        )

    def _calc_ai_score(self, ai_signal: AICandidate) -> float:
        sentiment_component = (ai_signal.sentiment_score + 1) * 50
        strength_component = ai_signal.event_strength * 100
        source_component = ai_signal.source_confidence * 100
        confidence_component = ai_signal.ai_confidence * 100

        score = (
            sentiment_component * 0.25
            + strength_component * 0.35
            + source_component * 0.25
            + confidence_component * 0.15
        )

        # TODO:
        # - 根据 risk_flags 扣分
        # - 根据 source_urls 多源验证加分
        # - 根据 event_type 调权
        return max(0, min(100, score))

    def _calc_quant_score(self, quant_signal: QuantResult) -> float:
        score = (
            quant_signal.trend_score * 0.35
            + quant_signal.momentum_score * 0.25
            + (100 - quant_signal.risk_score) * 0.25
            + (100 - quant_signal.overheat_score) * 0.15
        )
        return max(0, min(100, score))

    def _ai_view(self, ai_signal: AICandidate):
        if ai_signal.sentiment_score > 0.3 and ai_signal.event_strength > 0.5:
            return "positive"
        if ai_signal.sentiment_score < -0.3:
            return "negative"
        if ai_signal.ai_confidence < 0.4:
            return "uncertain"
        return "neutral"

    def _quant_view(self, quant_signal: QuantResult):
        if quant_signal.quant_decision in {DecisionLevel.STRONG_WATCH, DecisionLevel.WATCH}:
            return "support"
        if quant_signal.quant_decision == DecisionLevel.AVOID:
            return "reject"
        if quant_signal.quant_decision == DecisionLevel.RISKY:
            return "uncertain"
        return "neutral"
