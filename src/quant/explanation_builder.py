from __future__ import annotations

from src.quant.feature_schema import QuantFeatureVector, QuantScoreBreakdown


class ExplanationBuilder:
    """根据已计算的特征和分数生成中文解释，不参与评分。"""

    def build(
        self,
        features: QuantFeatureVector,
        scores: QuantScoreBreakdown,
    ) -> tuple[str, list[str]]:
        reasons: list[str] = []
        tags: list[str] = []

        def add(reason: str, tag: str) -> None:
            reasons.append(reason)
            tags.append(tag)

        if features.data_quality_score < 50:
            add("数据质量不足，部分指标可靠性较低。", "data_quality_low")
        if features.is_latest_suspended:
            add("最新交易状态异常，可能存在停牌或不可交易风险。", "suspended")
        if features.is_st is True:
            add("该股票被标记为 ST，风险较高。", "st_risk")

        if scores.trend_score >= 70:
            add("均线结构偏强，短中期趋势较好。", "trend_strong")
        elif scores.trend_score <= 40:
            add("均线结构偏弱，价格未能站稳关键均线。", "trend_weak")

        if scores.momentum_score >= 70:
            add("近期收益表现较强，存在一定动量。", "momentum_strong")
        elif scores.momentum_score <= 40:
            add("近期收益表现偏弱，短期动量不足。", "momentum_weak")

        if scores.volume_score >= 70:
            add("成交量或成交额出现温和放大，市场关注度有所提升。", "volume_confirmed")
        if features.volume_ratio_5d is not None and features.volume_ratio_5d > 4:
            add("成交量异常放大，可能存在情绪过热或短期冲高风险。", "abnormal_volume")

        if scores.risk_score >= 75:
            add("近期波动率、回撤或 ATR 偏高，风险较大。", "high_risk")
        if scores.overheat_score >= 75:
            add("短期涨幅、RSI 或布林带位置显示过热风险。", "overheated")

        if scores.macd_score >= 65:
            add("MACD 动量为正，短线趋势有一定支撑。", "macd_positive")
        elif scores.macd_score <= 45:
            add("MACD 动量偏弱，短线趋势支撑不足。", "macd_weak")

        if not reasons:
            reasons.append("量化指标整体中性，暂未发现明显趋势优势或极端风险。")
            tags.append("neutral")

        return "；".join(reasons), tags
