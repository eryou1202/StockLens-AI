from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class QuantFeatureVector(BaseModel):
    """Rule baseline v1.0 的完整内部特征向量。"""

    symbol: str
    provider: str = "unknown"
    frequency: str = "1d"
    adjust_type: str = "qfq"
    data_rows: int = 0
    data_quality_score: float = Field(default=20.0, ge=0, le=100)
    is_data_sufficient: bool = False
    is_latest_suspended: bool = False
    is_st: bool | None = None

    latest_open: float | None = None
    latest_high: float | None = None
    latest_low: float | None = None
    latest_close: float | None = None
    latest_pre_close: float | None = None
    latest_volume: float | None = None
    latest_amount: float | None = None
    latest_turnover_rate: float | None = None

    return_1d: float | None = None
    return_3d: float | None = None
    return_5d: float | None = None
    return_10d: float | None = None
    return_20d: float | None = None
    return_60d: float | None = None

    ma5: float | None = None
    ma10: float | None = None
    ma20: float | None = None
    ma60: float | None = None

    ma5_ma20_gap: float | None = None
    ma10_ma20_gap: float | None = None
    ma20_ma60_gap: float | None = None
    close_ma20_gap: float | None = None

    volume_ratio_5d: float | None = None
    volume_ratio_20d: float | None = None
    amount_ratio_5d: float | None = None
    amount_ratio_20d: float | None = None

    volatility_20d: float | None = None
    max_drawdown_20d: float | None = None
    atr_14: float | None = None

    rsi_14: float | None = None
    macd: float | None = None
    macd_signal: float | None = None
    macd_hist: float | None = None
    bollinger_position: float | None = None
    bollinger_percent_b: float | None = None

    warnings: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class QuantScoreBreakdown(BaseModel):
    """Rule baseline v1.0 的评分明细。"""

    trend_score: float = Field(ge=0, le=100)
    momentum_score: float = Field(ge=0, le=100)
    volume_score: float = Field(ge=0, le=100)
    risk_score: float = Field(ge=0, le=100)
    overheat_score: float = Field(ge=0, le=100)
    macd_score: float = Field(ge=0, le=100)
    support_score: float = Field(ge=0, le=100)
    penalty_score: float = Field(ge=0, le=100)
    quant_score: float = Field(ge=0, le=100)
    heuristic_prob_up_5d: float = Field(ge=0, le=1)
    quant_decision: Literal["support", "uncertain", "reject"]
    score_tags: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
