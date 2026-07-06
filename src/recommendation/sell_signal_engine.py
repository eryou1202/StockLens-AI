from __future__ import annotations

from src.models.schemas import FinalDecision
from src.portfolio.position_schema import Position, PositionStatus
from src.recommendation.reason_engine import ReasonEngine
from src.recommendation.recommendation_engine import RecommendationEngine
from src.recommendation.recommendation_schema import ActionLevel, Recommendation, RecommendationAction
from src.recommendation.sell_rules import calc_unrealized_return, is_trend_broken, should_reduce, should_stop_loss, should_take_profit


class SellSignalEngine:
    def build_sell_signal(self, position: Position, decision: FinalDecision) -> Recommendation:
        context = RecommendationEngine.extract_quant_context(decision)
        current_price = ReasonEngine.number(context.get("latest_close"))
        quant_decision = str(context.get("quant_decision"))
        final_level = str(getattr(decision.final_level, "value", decision.final_level))
        is_watch = position.status == PositionStatus.WATCH_ONLY or position.entry_price <= 0.01
        base_metadata = {
            "position_id": position.id, "position_status": position.status.value,
            "is_watch_only": is_watch, "entry_price": position.entry_price,
            "current_price": current_price, "unrealized_return": None, "triggered_rules": [],
            **context,
        }

        if current_price is None or current_price <= 0:
            return self._result(position, decision, RecommendationAction.RISK_WARNING,
                ActionLevel.MEDIUM, ["当前缺少可用最新价格，只能暂停收益判断并继续观察。"],
                ["请先刷新行情数据；缺失价格时不计算浮动收益或止损止盈。"], base_metadata)

        if is_watch:
            return self._watch_only_signal(position, decision, base_metadata)

        unrealized = calc_unrealized_return(current_price, position.entry_price)
        base_metadata["unrealized_return"] = unrealized
        risk = float(context.get("risk_score") or 0.0)
        heat = float(context.get("overheat_score") or 0.0)
        rsi = ReasonEngine.number(context.get("rsi_14"))
        gap = ReasonEngine.number(context.get("close_ma20_gap"))
        macd = ReasonEngine.number(context.get("macd_hist"))
        momentum = ReasonEngine.number(context.get("momentum_score"))
        volume = ReasonEngine.number(context.get("volume_ratio_5d"))
        reasons: list[str] = []
        risks: list[str] = []
        rules: list[str] = []

        price_stop = position.stop_loss_price is not None and current_price <= position.stop_loss_price
        if should_stop_loss(unrealized) or price_stop:
            action, level = RecommendationAction.STOP_LOSS, ActionLevel.CRITICAL
            rules.append("position_stop_loss" if price_stop else "return_lte_-5pct")
            reasons.extend([
                "当前亏损已经触及止损条件，需要优先关注风险是否继续扩大。",
                f"当前价格为 {current_price:.2f} 元，成本价为 {position.entry_price:.2f} 元，"
                f"浮动收益为 {unrealized:.2%}，已低于 -5.00% 的止损参考线。",
            ])
            if price_stop:
                reasons.append(f"当前价格也已低于自定义止损价 {position.stop_loss_price:.2f} 元，应重点检查原持仓逻辑是否仍然成立。")
        elif should_take_profit(unrealized, heat):
            action, level = RecommendationAction.TAKE_PROFIT, ActionLevel.HIGH
            rules.extend(["return_gte_12pct", "overheat_gte_70"])
            reasons.extend([
                "当前已有较高浮盈，同时短期走势开始偏热，可以考虑保护已有收益。",
                f"当前浮动收益为 {unrealized:.2%}，已达到 12.00% 的止盈观察线。",
                f"短期过热评分为 {heat:.2f} 分，已达到 70 分的止盈辅助线，继续持有需防范回落。",
            ])
            reasons.extend(self._heat_details(context))
        elif should_reduce(unrealized, rsi, heat):
            action = RecommendationAction.REDUCE
            level = ActionLevel.HIGH if unrealized >= 0.15 or heat >= 85 else ActionLevel.MEDIUM
            rules.extend(["return_gte_8pct", "rsi_gte_75", "overheat_gte_65"])
            reasons.extend([
                "当前已有一定浮盈，但短期指标偏热，可以考虑适度降低风险敞口。",
                f"当前浮动收益为 {unrealized:.2%}，高于 8.00% 的减仓观察线。",
                f"RSI 相对强弱指标为 {rsi:.2f}，高于 75；短期过热评分为 {heat:.2f} 分，高于 65 分参考线。",
            ])
        elif is_trend_broken(gap, macd, momentum) or (quant_decision == "reject" and final_level in {"risky", "avoid"}):
            action, level = RecommendationAction.SELL_ALERT, ActionLevel.HIGH
            if is_trend_broken(gap, macd, momentum):
                rules.append("trend_broken")
            if quant_decision == "reject":
                rules.append("quant_reject")
            reasons.append("当前趋势和动量都偏弱，持仓支撑下降，需要重点观察是否继续走弱。")
            if quant_decision == "reject":
                reasons.append("量化判断当前不支持继续积极持有，综合状态已进入风险或回避区域。")
            reasons.extend(ReasonEngine.weakness_details(context))
            reasons.extend(ReasonEngine.diagnostic_snapshot(context))
        elif self._risk_triggered(context):
            action, level = RecommendationAction.RISK_WARNING, ActionLevel.MEDIUM
            rules.append("risk_or_overheat")
            risks.extend(ReasonEngine.weakness_details(context))
            reasons.append("当前出现明显的风险或过热信号，适合先控制预期并重点观察，而不是盲目加仓。")
        else:
            action, level = RecommendationAction.HOLD, ActionLevel.LOW
            rules.append("no_exit_rule")
            reasons.extend([
                "当前没有触发止损、止盈或趋势破坏条件，可以继续观察，但仍需关注趋势和风险变化。",
                f"当前浮动收益为 {unrealized:.2%}，尚未达到预设的止损或止盈参考线。",
                ReasonEngine.score_observation("trend_score", context.get("trend_score"), 70),
                ReasonEngine.score_observation("momentum_score", context.get("momentum_score"), 40),
                ReasonEngine.score_observation("risk_score", context.get("risk_score"), 75),
                ReasonEngine.score_observation("overheat_score", context.get("overheat_score"), 75),
            ])

        base_metadata["triggered_rules"] = rules
        base_metadata["triggered_rule_labels"] = self._rule_labels(rules)
        risks.extend(ReasonEngine.weakness_details(context))
        return self._result(position, decision, action, level, [x for x in reasons if x],
                            list(dict.fromkeys(risks)), base_metadata)

    def _watch_only_signal(self, position: Position, decision: FinalDecision, metadata: dict) -> Recommendation:
        context = metadata
        weak = ReasonEngine.weakness_details(context)
        if context.get("quant_decision") == "reject" or weak:
            action, level = RecommendationAction.RISK_WARNING, ActionLevel.MEDIUM
            metadata["triggered_rules"] = ["watch_only_quant_risk"]
            reasons = ["这是观察股，不是真实持仓；系统只做量化跟踪，不计算止损、止盈或浮动收益。"] + weak
        else:
            action, level = RecommendationAction.HOLD, ActionLevel.LOW
            metadata["triggered_rules"] = ["watch_only_observation"]
            reasons = [
                "这是观察股，不是真实持仓；系统只做量化跟踪，不计算止损、止盈或浮动收益。",
                ReasonEngine.score_observation("trend_score", context.get("trend_score"), 70),
                ReasonEngine.score_observation("momentum_score", context.get("momentum_score"), 40),
                ReasonEngine.score_observation("risk_score", context.get("risk_score"), 75),
                ReasonEngine.score_observation("overheat_score", context.get("overheat_score"), 75),
            ]
        metadata["triggered_rule_labels"] = self._rule_labels(metadata["triggered_rules"])
        return self._result(position, decision, action, level, [x for x in reasons if x], weak, metadata)

    @staticmethod
    def _risk_triggered(context: dict) -> bool:
        checks = [
            ("overheat_score", 75, "ge"), ("risk_score", 75, "ge"),
            ("volume_ratio_5d", 3, "gt"), ("rsi_14", 75, "ge"),
            ("bollinger_position", 1.2, "ge"), ("max_drawdown_20d", -0.12, "le"),
            ("volatility_20d", 0.04, "ge"),
        ]
        for name, threshold, op in checks:
            value = ReasonEngine.number(context.get(name))
            if value is not None and ((op == "ge" and value >= threshold) or (op == "gt" and value > threshold) or (op == "le" and value <= threshold)):
                return True
        return False

    @staticmethod
    def _heat_details(context: dict) -> list[str]:
        details: list[str] = []
        for key, name, threshold, percent in (
            ("rsi_14", "RSI 相对强弱指标", 75, False),
            ("bollinger_position", "布林带位置", 1.2, False),
            ("return_5d", "近 5 日涨跌幅", 0.05, True),
        ):
            value = ReasonEngine.number(context.get(key))
            if value is not None:
                rendered = f"{value:.2%}" if percent else f"{value:.2f}"
                limit = f"{threshold:.2%}" if percent else f"{threshold:.2f}"
                details.append(f"{name}当前为 {rendered}，过热参考线为 {limit}；数值越高，短线回落风险通常越值得关注。")
        return details

    @staticmethod
    def _rule_labels(rules: list[str]) -> list[str]:
        labels = {
            "position_stop_loss": "跌破自定义止损价",
            "return_lte_-5pct": "浮亏达到 5% 止损线",
            "return_gte_12pct": "浮盈达到 12% 止盈观察线",
            "overheat_gte_70": "短期过热评分达到 70 分",
            "return_gte_8pct": "浮盈达到 8% 减仓观察线",
            "rsi_gte_75": "RSI 指标达到 75",
            "overheat_gte_65": "短期过热评分达到 65 分",
            "trend_broken": "趋势与动量同时转弱",
            "quant_reject": "量化判断不支持继续积极持有",
            "risk_or_overheat": "风险或过热指标达到警戒线",
            "no_exit_rule": "尚未触发退出条件",
            "watch_only_quant_risk": "观察股量化风险偏高",
            "watch_only_observation": "观察股常规量化跟踪",
        }
        return [labels.get(rule, "其他观察条件") for rule in rules]

    @staticmethod
    def _result(position: Position, decision: FinalDecision, action: RecommendationAction,
                level: ActionLevel, reasons: list[str], risks: list[str], metadata: dict) -> Recommendation:
        confidence = {
            RecommendationAction.STOP_LOSS: .95, RecommendationAction.TAKE_PROFIT: .85,
            RecommendationAction.REDUCE: .75, RecommendationAction.SELL_ALERT: .75,
            RecommendationAction.RISK_WARNING: .65, RecommendationAction.HOLD: .55,
        }[action]
        return Recommendation(
            symbol=position.symbol, stock_name=position.stock_name or decision.stock_name,
            source_type=str(position.metadata.get("source_type", "position")),
            as_of_time=decision.as_of_time, action=action, action_level=level,
            confidence=confidence, reason=reasons, risks=risks,
            invalid_conditions=["关注股价是否跌破 20 日均线，以及 MACD 动能柱和短期动量是否继续转弱。"],
            ai_view=decision.ai_view, quant_decision=str(metadata.get("quant_decision")),
            final_level=str(getattr(decision.final_level, "value", decision.final_level)),
            final_score=decision.final_score, metadata=metadata,
        )
