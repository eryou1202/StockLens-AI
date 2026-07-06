from __future__ import annotations

from typing import Any


class ReasonEngine:
    """Turns internal quant fields into beginner-friendly Chinese explanations."""

    INDICATOR_NAMES = {
        "latest_close": "最新价格",
        "close_ma20_gap": "股价相对 20 日均线",
        "ma5_ma20_gap": "5 日均线相对 20 日均线",
        "ma20_ma60_gap": "20 日均线相对 60 日均线",
        "macd_hist": "MACD 动能柱",
        "momentum_score": "短期动量评分",
        "trend_score": "趋势评分",
        "risk_score": "综合风险评分",
        "overheat_score": "短期过热评分",
        "rsi_14": "RSI 相对强弱指标",
        "volume_ratio_5d": "5 日成交量放大倍数",
        "amount_ratio_5d": "5 日成交额放大倍数",
        "max_drawdown_20d": "近 20 日最大回撤",
        "volatility_20d": "近 20 日波动率",
        "atr_14": "ATR 波动指标",
        "bollinger_position": "布林带位置",
        "return_5d": "近 5 日涨跌幅",
        "return_20d": "近 20 日涨跌幅",
        "quant_score": "综合量化评分",
        "volume_score": "成交量评分",
        "macd_score": "MACD 评分",
    }

    @classmethod
    def display_name(cls, field: str) -> str:
        return cls.INDICATOR_NAMES.get(field, field)

    @staticmethod
    def number(value: Any) -> float | None:
        try:
            return None if value is None else float(value)
        except (TypeError, ValueError, OverflowError):
            return None

    @classmethod
    def support_details(cls, context: dict[str, Any]) -> list[str]:
        result: list[str] = []
        quant = cls.number(context.get("quant_score"))
        trend = cls.number(context.get("trend_score"))
        momentum = cls.number(context.get("momentum_score"))
        risk = cls.number(context.get("risk_score"))
        heat = cls.number(context.get("overheat_score"))
        if quant is not None:
            result.append(
                f"综合量化评分为 {quant:.2f} 分，参考支持线为 65 分；"
                + ("目前量价条件整体偏积极。" if quant >= 65 else "目前还没有形成足够明确的量化优势。")
            )
        if trend is not None:
            result.append(
                f"趋势评分为 {trend:.2f} 分，70 分以上通常代表趋势较强；"
                + ("当前均线结构对后续观察有一定支撑。" if trend >= 70 else "当前趋势尚未达到强势区，需要继续确认。")
            )
        if momentum is not None:
            result.append(
                f"短期动量评分为 {momentum:.2f} 分，70 分以上代表上涨力量较强；"
                + ("最近价格表现具有一定延续性。" if momentum >= 70 else "上涨力量仍属一般，不宜只凭短期波动追入。")
            )
        if risk is not None:
            result.append(
                f"综合风险评分为 {risk:.2f} 分，75 分是高风险参考线；"
                + ("当前风险尚未进入高位，但仍需设置观察条件。" if risk < 75 else "当前波动或回撤风险偏高，应降低预期。")
            )
        if heat is not None:
            result.append(
                f"短期过热评分为 {heat:.2f} 分，75 分以上需要警惕追高；"
                + ("目前没有明显过热。" if heat < 75 else "当前价格可能已经偏热，追入风险增加。")
            )
        return result

    @classmethod
    def diagnostic_snapshot(cls, context: dict[str, Any]) -> list[str]:
        result: list[str] = []
        gap = cls.number(context.get("close_ma20_gap"))
        macd = cls.number(context.get("macd_hist"))
        momentum = cls.number(context.get("momentum_score"))
        risk = cls.number(context.get("risk_score"))
        heat = cls.number(context.get("overheat_score"))
        if gap is not None:
            if gap < 0:
                result.append(f"股价目前低于 20 日均线约 {abs(gap):.2%}，说明短期走势还没有重新站稳。")
            else:
                result.append(f"股价目前高于 20 日均线约 {gap:.2%}，短期趋势仍有一定支撑。")
        if macd is not None:
            result.append(
                f"MACD 动能柱当前为 {macd:.4f}，以 0 为强弱参考线；"
                + ("动能仍偏弱，需要观察是否重新转正。" if macd < 0 else "短线动能保持正向，但仍要结合价格确认。")
            )
        if momentum is not None:
            result.append(
                f"短期动量评分为 {momentum:.2f} 分，40 分是偏弱参考线；"
                + ("最近上涨力量不足。" if momentum < 40 else "当前动量暂未进入明显弱势区。")
            )
        if risk is not None:
            result.append(
                f"综合风险评分为 {risk:.2f} 分，75 分是高风险参考线；"
                + ("当前风险偏高，需要控制仓位和预期。" if risk >= 75 else "当前尚未达到高风险线。")
            )
        if heat is not None:
            result.append(
                f"短期过热评分为 {heat:.2f} 分，75 分以上要防止追高；"
                + ("当前存在过热迹象。" if heat >= 75 else "当前未达到明显过热线。")
            )
        return result

    @classmethod
    def weakness_details(cls, context: dict[str, Any]) -> list[str]:
        result: list[str] = []
        gap = cls.number(context.get("close_ma20_gap"))
        macd = cls.number(context.get("macd_hist"))
        momentum = cls.number(context.get("momentum_score"))
        risk = cls.number(context.get("risk_score"))
        heat = cls.number(context.get("overheat_score"))
        rsi = cls.number(context.get("rsi_14"))
        volume = cls.number(context.get("volume_ratio_5d"))
        amount = cls.number(context.get("amount_ratio_5d"))
        bollinger = cls.number(context.get("bollinger_position"))
        drawdown = cls.number(context.get("max_drawdown_20d"))
        volatility = cls.number(context.get("volatility_20d"))
        atr = cls.number(context.get("atr_14"))
        return_5d = cls.number(context.get("return_5d"))
        return_20d = cls.number(context.get("return_20d"))

        if gap is not None and gap < 0:
            result.append(f"股价目前低于 20 日均线约 {abs(gap):.2%}，短期走势尚未站稳，持仓或观察都应更谨慎。")
        if macd is not None and macd < 0:
            result.append(f"MACD 动能柱为 {macd:.4f}，低于 0 轴，说明短线上涨动能偏弱。")
        if momentum is not None and momentum < 40:
            result.append(f"短期动量评分为 {momentum:.2f} 分，低于 40 分，说明最近上涨力量不足。")
        if risk is not None and risk >= 75:
            result.append(f"综合风险评分为 {risk:.2f} 分，已达到 75 分高风险线，当前更适合控制仓位或继续观察。")
        if heat is not None and heat >= 75:
            result.append(f"短期过热评分为 {heat:.2f} 分，已达到 75 分警戒线，继续追高容易承受回落。")
        if rsi is not None and rsi >= 75:
            result.append(f"RSI 相对强弱指标为 {rsi:.2f}，高于 75，短期买盘可能过于集中，需要防范冲高回落。")
        if volume is not None and volume > 3:
            result.append(f"5 日成交量放大倍数为 {volume:.2f} 倍，高于 3 倍参考线；放量较急，需要确认价格能否同步走强。")
        if amount is not None and amount > 3:
            result.append(f"5 日成交额放大倍数为 {amount:.2f} 倍，高于 3 倍参考线，资金活跃度明显上升，也要留意短线情绪过热。")
        if bollinger is not None and bollinger >= 1.2:
            result.append(f"布林带位置为 {bollinger:.2f}，高于 1.20 的过热参考线，价格可能已经偏离正常波动区间。")
        if drawdown is not None and drawdown <= -0.12:
            result.append(f"近 20 日最大回撤约 {abs(drawdown):.2%}，超过 12% 的风险参考线，说明近期回落幅度较大。")
        if volatility is not None and volatility >= 0.04:
            result.append(f"近 20 日波动率约 {volatility:.2%}，高于 4% 的参考线，价格起伏较大，新手应避免忽视回撤。")
        if atr is not None and atr >= 0.08:
            result.append(f"ATR 波动指标约为 {atr:.2%}，高于 8% 的偏高参考线，单日价格波动可能较大。")
        if return_5d is not None and return_5d >= 0.15:
            result.append(f"近 5 日累计上涨约 {return_5d:.2%}，已达到 15% 的过热参考线，短线继续追入的性价比下降。")
        if return_20d is not None and return_20d >= 0.40:
            result.append(f"近 20 日累计上涨约 {return_20d:.2%}，已达到 40% 的过热参考线，需要防范获利回吐。")
        return result

    @classmethod
    def score_observation(cls, field: str, value: Any, threshold: float) -> str:
        number = cls.number(value)
        name = cls.display_name(field)
        if number is None:
            return f"{name}暂时缺少数据，本次按中性状态观察。"
        if field in {"risk_score", "overheat_score"}:
            meaning = "已达到警戒线，需要提高风险意识" if number >= threshold else "尚未达到警戒线"
        elif field == "momentum_score":
            meaning = "上涨力量偏弱" if number < threshold else "上涨力量暂未明显转弱"
        else:
            meaning = "趋势较强" if number >= threshold else "趋势仍需确认"
        return f"{name}为 {number:.2f} 分，参考线为 {threshold:.0f} 分，{meaning}。"
