from __future__ import annotations

from src.models.schemas import FinalDecision
from src.portfolio.position_schema import Position
from src.recommendation.recommendation_engine import RecommendationEngine
from src.recommendation.recommendation_schema import (
    ActionLevel,
    Recommendation,
    RecommendationAction,
)
from src.recommendation.sell_rules import (
    calc_unrealized_return,
    is_trend_broken,
    should_reduce,
    should_stop_loss,
    should_take_profit,
)


class SellSignalEngine:
    def build_sell_signal(
        self,
        position: Position,
        decision: FinalDecision,
    ) -> Recommendation:
        context = RecommendationEngine.extract_quant_context(decision)
        current_price = context.get("latest_close")
        quant_decision = context.get("quant_decision")
        final_level = str(getattr(decision.final_level, "value", decision.final_level))

        if current_price is None or float(current_price) <= 0:
            return Recommendation(
                symbol=position.symbol,
                stock_name=position.stock_name,
                as_of_time=decision.as_of_time,
                action=RecommendationAction.RISK_WARNING,
                action_level=ActionLevel.MEDIUM,
                confidence=0.3,
                reason=["缺少可用的最新价格，暂时无法计算持仓收益和卖出提醒。"],
                risks=["请先刷新行情数据后再检查持仓。"],
                ai_view=decision.ai_view,
                quant_decision=quant_decision,
                final_level=final_level,
                final_score=decision.final_score,
                metadata={"entry_price": position.entry_price, **context},
            )

        current_price = float(current_price)
        unrealized_return = calc_unrealized_return(current_price, position.entry_price)
        risk_score = float(context.get("risk_score") or 0.0)
        overheat_score = float(context.get("overheat_score") or 0.0)
        rsi_14 = context.get("rsi_14")
        close_ma20_gap = context.get("close_ma20_gap")
        macd_hist = context.get("macd_hist")
        momentum_score = context.get("momentum_score")
        volume_ratio = context.get("volume_ratio_5d")
        reasons: list[str] = []
        risks: list[str] = []

        price_stop_triggered = (
            position.stop_loss_price is not None and current_price <= position.stop_loss_price
        )
        if should_stop_loss(unrealized_return) or price_stop_triggered:
            action = RecommendationAction.STOP_LOSS
            level = ActionLevel.CRITICAL
            reasons.append("当前持仓收益已触发止损阈值。")
            if price_stop_triggered:
                reasons.append("当前价格已低于持仓设置的止损价。")
        elif should_take_profit(unrealized_return, overheat_score):
            action = RecommendationAction.TAKE_PROFIT
            level = ActionLevel.HIGH
            reasons.append("当前浮盈较高且量化过热分偏高，触发止盈提醒。")
        elif should_reduce(unrealized_return, rsi_14, overheat_score):
            action = RecommendationAction.REDUCE
            level = ActionLevel.HIGH if unrealized_return >= 0.15 or overheat_score >= 85 else ActionLevel.MEDIUM
            reasons.append("当前已有一定浮盈，RSI 与过热分提示可考虑降低风险敞口。")
        elif is_trend_broken(close_ma20_gap, macd_hist, momentum_score) or (
            quant_decision == "reject" and final_level in {"risky", "avoid"}
        ):
            action = RecommendationAction.SELL_ALERT
            level = ActionLevel.HIGH
            reasons.append("趋势、MACD 或动量显示持仓支撑转弱，触发卖出观察提醒。")
        elif (
            overheat_score >= 75
            or risk_score >= 75
            or (volume_ratio is not None and float(volume_ratio) > 3)
        ):
            action = RecommendationAction.RISK_WARNING
            level = ActionLevel.MEDIUM
            reasons.append("当前持仓出现风险、过热或异常放量信号。")
        else:
            action = RecommendationAction.HOLD
            level = ActionLevel.LOW
            reasons.append("当前未触发止损、止盈、减仓或趋势破坏条件，继续观察持有。")

        if risk_score >= 75:
            risks.append("量化风险分偏高。")
        if overheat_score >= 75:
            risks.append("短期过热分偏高。")
        if close_ma20_gap is not None and float(close_ma20_gap) < 0:
            risks.append("价格位于 MA20 下方。")

        confidence = {
            RecommendationAction.STOP_LOSS: 0.95,
            RecommendationAction.TAKE_PROFIT: 0.85,
            RecommendationAction.REDUCE: 0.75,
            RecommendationAction.SELL_ALERT: 0.75,
            RecommendationAction.RISK_WARNING: 0.65,
            RecommendationAction.HOLD: 0.55,
        }[action]
        return Recommendation(
            symbol=position.symbol,
            stock_name=position.stock_name or decision.stock_name,
            as_of_time=decision.as_of_time,
            action=action,
            action_level=level,
            confidence=confidence,
            reason=reasons,
            risks=risks,
            invalid_conditions=["行情或量化条件发生变化时重新检查。"],
            ai_view=decision.ai_view,
            quant_decision=quant_decision,
            final_level=final_level,
            final_score=decision.final_score,
            metadata={
                "position_id": position.id,
                "entry_price": position.entry_price,
                "current_price": current_price,
                "unrealized_return": unrealized_return,
                **context,
            },
        )
