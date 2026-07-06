from __future__ import annotations

import re

from src.models.schemas import FinalDecision
from src.recommendation.reason_engine import ReasonEngine
from src.recommendation.recommendation_schema import Recommendation, RecommendationAction


ACTION_LABELS = {
    RecommendationAction.BUY_CANDIDATE: "🟢 买入候选",
    RecommendationAction.WATCH: "🔵 观察",
    RecommendationAction.AVOID: "⚠️ 避开",
    RecommendationAction.RISK_WARNING: "🟠 风险预警",
    RecommendationAction.HOLD: "🟢 持有",
    RecommendationAction.REDUCE: "🟡 减仓提醒",
    RecommendationAction.TAKE_PROFIT: "🟡 止盈提醒",
    RecommendationAction.SELL_ALERT: "🔴 卖出提醒",
    RecommendationAction.STOP_LOSS: "🔴 止损",
}

RISK_TYPE_NAMES = {
    "already_priced_in": "利好可能已反映在价格中",
    "overheated": "短期过热",
    "rumor_only": "信息仍停留在传闻阶段",
    "weak_source": "信息来源可靠性有限",
    "conflicting_info": "不同信息之间存在冲突",
    "one_time_gain": "收益可能属于一次性因素",
    "valuation_pressure": "估值压力",
    "liquidity_risk": "流动性风险",
    "policy_uncertainty": "政策不确定性",
    "earnings_uncertainty": "业绩不确定性",
    "negative_event": "负面事件",
    "unknown": "其他风险",
}


class RecommendationExplainer:
    @classmethod
    def explain(cls, decision: FinalDecision, recommendation: Recommendation) -> Recommendation:
        reasons: list[str] = [cls._summary(decision, recommendation)]
        risks: list[str] = []
        context = recommendation.metadata
        manual = recommendation.source_type == "manual_watch"

        if manual:
            reasons.append("这只股票由用户手动加入观察池，不代表 AI 信息面推荐。")
            if recommendation.quant_decision == "support":
                reasons.append("量化状态偏积极，但缺少明确的信息面催化，因此暂不列入买入候选。")
                reasons.extend(ReasonEngine.support_details(context))
            elif recommendation.quant_decision == "uncertain":
                reasons.append("量化状态暂不明确，建议等待趋势和成交量出现更清晰的确认。")
            else:
                reasons.append("当前量价条件偏弱，人工关注理由不足以抵消趋势或风险压力。")
                risks.extend(ReasonEngine.weakness_details(context))
        elif recommendation.action == RecommendationAction.BUY_CANDIDATE:
            reasons.append("AI 信息面偏正面，量化状态也达到支持条件，两方面暂时同向。")
            reasons.extend(ReasonEngine.support_details(context))
        elif recommendation.action in {RecommendationAction.RISK_WARNING, RecommendationAction.AVOID}:
            if recommendation.quant_decision == "reject":
                reasons.append("当前规则量化没有通过，趋势、动量或风险条件中至少有一项不够理想。")
            if decision.ai_view == "negative":
                reasons.append("信息面方向偏负面，因此当前不适合作为积极买入候选。")
            risks.extend(ReasonEngine.weakness_details(context))
            reasons.extend(ReasonEngine.diagnostic_snapshot(context))
        else:
            if decision.ai_view == "positive" and recommendation.quant_decision == "uncertain":
                reasons.append("信息面虽然偏正面，但量价走势尚未确认，不宜仅凭消息追入。")
            elif decision.ai_view == "neutral" and recommendation.quant_decision == "support":
                reasons.append("量化走势相对较好，但信息面缺少明确催化，因此先观察而不是直接追入。")
                reasons.extend(ReasonEngine.support_details(context))
            else:
                reasons.append("信息面与量化条件尚未同时达到较强状态，继续观察更稳妥。")

        if not risks and recommendation.action in {RecommendationAction.RISK_WARNING, RecommendationAction.AVOID}:
            risks.append("综合条件尚未通过当前推荐规则，建议等待价格和动量进一步改善。")
        for item in decision.quant_result.quant_reason[:2]:
            reasons.append(cls._humanize_text(item))
        risks.extend(cls._humanize_ai_risk(item) for item in decision.ai_candidate.risk_flags[:3])

        return recommendation.model_copy(update={
            "reason": cls._deduplicate(reasons),
            "risks": cls._deduplicate(risks),
            "invalid_conditions": cls._invalid_conditions(),
        })

    @staticmethod
    def _summary(decision: FinalDecision, item: Recommendation) -> str:
        if item.source_type == "manual_watch":
            if item.quant_decision == "reject":
                return "当前量价状态偏弱，这只人工观察股需要谨慎跟踪，暂不适合直接追入。"
            return "这是一只人工观察股；量化结果只用于辅助跟踪，不会直接把它列为买入候选。"
        summaries = {
            RecommendationAction.BUY_CANDIDATE: "信息面和量化条件暂时形成配合，可列入重点候选，但仍需等待实际价格确认。",
            RecommendationAction.WATCH: "当前更适合作为观察股，等待趋势、动量或信息面进一步确认。",
            RecommendationAction.RISK_WARNING: "当前有一定关注理由，但量化条件不支持直接追入，风险优先于机会。",
            RecommendationAction.AVOID: "当前趋势或风险条件不理想，暂时避开比贸然参与更稳妥。",
        }
        return summaries.get(item.action, "当前信号仍需继续观察，不构成确定的交易结论。")

    @staticmethod
    def _invalid_conditions() -> list[str]:
        return [
            "关注股价能否重新站上并稳住 20 日均线。",
            "关注 MACD 动能柱能否重新转为正值，确认短线力量恢复。",
            "放量时价格也应同步走强；若只放量不涨，需要警惕冲高回落。",
            "综合风险或短期过热评分达到 75 分时，应重新评估风险。",
        ]

    @classmethod
    def _humanize_ai_risk(cls, text: str) -> str:
        value = str(text)
        match = re.match(r"^\[[^]]+\]\s+([^:]+):\s*(.*)$", value)
        if not match:
            return cls._humanize_text(value)
        risk_type, detail = match.groups()
        label = RISK_TYPE_NAMES.get(risk_type, "信息面风险")
        return f"{label}：{detail}"

    @classmethod
    def _humanize_text(cls, text: str) -> str:
        value = str(text)
        for field, label in ReasonEngine.INDICATOR_NAMES.items():
            value = value.replace(field, label)
        value = value.replace("MA20", "20 日均线").replace("MACD hist", "MACD 动能柱")
        return value

    @staticmethod
    def format_report(item: Recommendation) -> str:
        conclusion = item.reason[0] if item.reason else "当前信息不足，建议继续观察。"
        main_reasons = item.reason[1:] if len(item.reason) > 1 else []
        lines = [
            f"{item.symbol} {item.stock_name or '未知名称'}", "",
            f"来源：{item.source_type}", f"动作：{ACTION_LABELS[item.action]}",
            f"等级：{item.action_level.value}", f"信心：{item.confidence:.2%}",
            f"AI 观点：{item.ai_view or '未知'}", f"量化观点：{item.quant_decision or '未知'}",
            f"最终分数：{'未知' if item.final_score is None else f'{item.final_score:.2f}'}", "",
            "结论：", conclusion, "",
            "主要原因：", *([f"- {value}" for value in main_reasons] or ["- 暂无更多原因说明。"]), "",
            "风险点：", *([f"- {value}" for value in item.risks] or ["- 暂未发现额外的高风险提示。"]), "",
            "后续观察点：", *[f"- {value}" for value in item.invalid_conditions],
        ]
        return "\n".join(lines)

    @staticmethod
    def _deduplicate(items) -> list[str]:
        return list(dict.fromkeys(str(item) for item in items if item))
