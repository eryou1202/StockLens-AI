from __future__ import annotations

from src.models.schemas import FinalDecision
from src.recommendation.recommendation_schema import Recommendation, RecommendationAction


ACTION_LABELS = {
    RecommendationAction.BUY_CANDIDATE: "🟢 买入候选",
    RecommendationAction.WATCH: "🔵 观察",
    RecommendationAction.AVOID: "⚪ 避开",
    RecommendationAction.RISK_WARNING: "🟠 风险预警",
    RecommendationAction.HOLD: "🟢 持有",
    RecommendationAction.REDUCE: "🟡 减仓提醒",
    RecommendationAction.TAKE_PROFIT: "🟣 止盈提醒",
    RecommendationAction.SELL_ALERT: "🔴 卖出提醒",
    RecommendationAction.STOP_LOSS: "🔴 止损",
}


class RecommendationExplainer:
    @staticmethod
    def explain(decision: FinalDecision, recommendation: Recommendation) -> Recommendation:
        reasons: list[str] = []
        risks: list[str] = []
        invalid_conditions: list[str] = []
        action = recommendation.action

        if action == RecommendationAction.BUY_CANDIDATE:
            reasons.extend([
                "AI 信息偏正面。",
                "量化判断为 support。",
                "趋势和动量对当前候选有一定支持。",
                "风险和过热分数未达到极端水平。",
            ])
            invalid_conditions.extend([
                "跌破 MA20",
                "MACD 动量继续转弱",
                "成交量放大但价格不上涨",
                "风险分或过热分继续升高",
            ])
        elif action == RecommendationAction.RISK_WARNING:
            reasons.extend([
                "AI 信息偏正面，但量化判断不支持当前追入。",
                "可能存在趋势偏弱、动量不足或消息已经反映在价格中的风险。",
            ])
            risks.extend(["短期趋势未修复。", "消息可能已经提前反映在价格中。"])
            invalid_conditions.extend([
                "重新站上 MA20",
                "MACD hist 转正",
                "成交量温和放大且价格同步上涨",
            ])
        elif action == RecommendationAction.AVOID:
            reasons.append("当前量化风险较高或趋势不支持，不建议进入买入候选。")
            invalid_conditions.extend(["风险分明显下降", "趋势结构重新转强", "MACD 动量转正"])
        else:
            reasons.extend([
                "当前 AI 或量化有一定支持，但尚未形成强买入候选。",
                "建议继续观察未来 3 到 5 个交易日的量价确认。",
            ])
            invalid_conditions.extend(["重新站上 MA20", "MACD hist 转正", "成交量温和放大"])

        for item in decision.quant_result.quant_reason[:2]:
            reasons.append(item)
        risks.extend(decision.ai_candidate.risk_flags[:3])

        risk_score = float(recommendation.metadata.get("risk_score") or 0.0)
        overheat_score = float(recommendation.metadata.get("overheat_score") or 0.0)
        volume_ratio = recommendation.metadata.get("volume_ratio_5d")
        if risk_score >= 75:
            risks.append("量化风险分偏高。")
        if overheat_score >= 75:
            risks.append("短期过热风险偏高。")
        if volume_ratio is None or float(volume_ratio) < 0.8:
            risks.append("成交量确认不足。")

        return recommendation.model_copy(
            update={
                "reason": RecommendationExplainer._deduplicate(reasons),
                "risks": RecommendationExplainer._deduplicate(risks),
                "invalid_conditions": RecommendationExplainer._deduplicate(invalid_conditions),
            }
        )

    @staticmethod
    def format_report(recommendation: Recommendation) -> str:
        label = ACTION_LABELS[recommendation.action]
        lines = [
            f"{recommendation.symbol} {recommendation.stock_name or ''}".strip(),
            "",
            f"动作：{label}",
            f"等级：{recommendation.action_level.value}",
            f"信心：{recommendation.confidence:.2f}",
            f"AI观点：{recommendation.ai_view or '-'}",
            f"量化观点：{recommendation.quant_decision or '-'}",
            f"最终分数：{'-' if recommendation.final_score is None else f'{recommendation.final_score:.1f}'}",
            "",
            "推荐结论：",
            recommendation.reason[0] if recommendation.reason else "当前信息不足，建议继续观察。",
            "",
            "主要原因：",
            *[f"- {item}" for item in recommendation.reason],
            "",
            "主要风险：",
            *([f"- {item}" for item in recommendation.risks] or ["- 暂无额外风险说明。"]),
            "",
            "触发升级或失效的条件：",
            *([f"- {item}" for item in recommendation.invalid_conditions] or ["- 继续观察量价变化。"]),
        ]
        return "\n".join(lines)

    @staticmethod
    def _deduplicate(items: list[str]) -> list[str]:
        return list(dict.fromkeys(item for item in items if item))
