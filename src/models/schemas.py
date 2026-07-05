from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class DecisionLevel(str, Enum):
    STRONG_WATCH = "strong_watch"
    WATCH = "watch"
    RISKY = "risky"
    AVOID = "avoid"


class AICandidate(BaseModel):
    """
    AI 推荐池中的单只股票。

    AI 只负责提供候选和信息面特征。
    量化模块会用 stock_code 去 MarketDataProvider 读取行情曲线。
    """

    stock_code: str = Field(..., description="内部统一代码，例如 000001.SZ / 600000.SH")
    stock_name: str | None = None
    as_of_time: datetime
    event_time: datetime | None = None

    event_type: str
    event_summary: str

    sentiment_score: float = Field(..., ge=-1, le=1)
    event_strength: float = Field(..., ge=0, le=1)
    source_confidence: float = Field(..., ge=0, le=1)
    ai_confidence: float = Field(..., ge=0, le=1)

    risk_flags: list[str] = Field(default_factory=list)
    source_urls: list[str] = Field(default_factory=list)
    raw_evidence: str | None = None

    metadata: dict[str, Any] = Field(default_factory=dict)


class MarketBar(BaseModel):
    """
    统一后的单根行情 K 线。

    无论原始数据来自 Baostock 还是 AKShare，最后都要转成这个结构。
    """

    symbol: str = Field(..., description="内部统一代码，例如 000001.SZ")
    trade_time: datetime
    frequency: str = Field(default="1d", description="1d / 5m / 15m / 30m / 60m")
    adjust_type: str = Field(default="qfq", description="none / qfq / hfq")

    open: float | None = None
    high: float | None = None
    low: float | None = None
    close: float | None = None
    pre_close: float | None = None

    volume: float | None = None
    amount: float | None = None
    turnover_rate: float | None = None
    pct_chg: float | None = None

    trade_status: str | None = None
    is_st: bool | None = None

    pe_ttm: float | None = None
    pb: float | None = None
    ps_ttm: float | None = None
    pcf_ncf_ttm: float | None = None

    provider: str = Field(default="unknown")
    raw_symbol: str | None = Field(default=None, description="数据源原始代码")
    fetched_at: datetime = Field(default_factory=datetime.now)

    raw: dict[str, Any] = Field(default_factory=dict)


class MarketDataBundle(BaseModel):
    """
    行情数据包。

    QuantEngine 只接受这个对象，不关心底层数据源。
    """

    symbol: str
    start_time: datetime
    end_time: datetime
    frequency: str = "1d"
    adjust_type: str = "qfq"
    provider: str = "unknown"

    bars: list[MarketBar] = Field(default_factory=list)

    data_quality: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def sorted_bars(self) -> list[MarketBar]:
        return sorted(self.bars, key=lambda x: x.trade_time)


class StockStatus(BaseModel):
    """
    股票状态。

    初版用于过滤停牌、ST 等不适合参与分析的股票。
    """

    symbol: str
    as_of_time: datetime
    is_trading: bool | None = None
    is_st: bool | None = None
    trade_status: str | None = None
    listed_date: datetime | None = None
    delisted_date: datetime | None = None
    provider: str = "unknown"
    raw: dict[str, Any] = Field(default_factory=dict)


class QuantFeatures(BaseModel):
    """
    量化特征。

    第一版由规则计算；后续可以直接作为 ML 输入。
    """

    symbol: str
    as_of_time: datetime

    return_1d: float | None = None
    return_5d: float | None = None
    return_20d: float | None = None

    ma5: float | None = None
    ma20: float | None = None
    ma60: float | None = None

    ma5_above_ma20: bool | None = None
    ma20_above_ma60: bool | None = None

    volume_ratio_5d: float | None = None
    volatility_20d: float | None = None
    max_drawdown_20d: float | None = None

    relative_strength_20d: float | None = None
    overheat_score: float | None = None
    liquidity_score: float | None = None

    extra: dict[str, Any] = Field(default_factory=dict)


class QuantResult(BaseModel):
    """
    量化曲线分析结果。
    """

    symbol: str
    as_of_time: datetime

    trend_state: Literal["uptrend", "downtrend", "sideways", "unknown"]
    volume_state: Literal["shrinking", "normal", "moderate_expansion", "abnormal_expansion", "unknown"]
    volatility_state: Literal["low", "medium", "high", "unknown"]

    momentum_score: float = Field(..., ge=0, le=100)
    trend_score: float = Field(..., ge=0, le=100)
    risk_score: float = Field(..., ge=0, le=100)
    overheat_score: float = Field(..., ge=0, le=100)

    prob_up_5d: float | None = Field(default=None, ge=0, le=1)
    prob_outperform_index_5d: float | None = Field(default=None, ge=0, le=1)
    predicted_return_5d: float | None = None
    max_drawdown_risk_5d: float | None = None

    model_confidence: float = Field(default=0.0, ge=0, le=1)
    quant_decision: DecisionLevel
    quant_reason: list[str] = Field(default_factory=list)

    features: QuantFeatures | None = None
    model_version: str = "rule_v0"


class FinalDecision(BaseModel):
    """
    判断层输出。
    """

    symbol: str
    stock_name: str | None = None
    as_of_time: datetime

    ai_view: Literal["positive", "negative", "neutral", "uncertain"]
    quant_view: Literal["support", "reject", "neutral", "uncertain"]

    final_level: DecisionLevel
    final_score: float = Field(..., ge=0, le=100)

    ai_summary: str
    quant_summary: str
    final_reason: list[str] = Field(default_factory=list)

    ai_candidate: AICandidate
    quant_result: QuantResult


class FeedbackResult(BaseModel):
    signal_id: str
    symbol: str
    as_of_time: datetime

    actual_return_1d: float | None = None
    actual_return_3d: float | None = None
    actual_return_5d: float | None = None
    actual_return_10d: float | None = None

    index_return_5d: float | None = None
    excess_return_5d: float | None = None
    max_drawdown_5d: float | None = None

    hit_5d: bool | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class FeedbackSummary(BaseModel):
    start_time: datetime
    end_time: datetime

    total_signals: int
    avg_return_5d: float | None = None
    avg_excess_return_5d: float | None = None
    hit_rate_5d: float | None = None

    event_type_performance: list[dict[str, Any]] = Field(default_factory=list)
    failure_patterns: list[str] = Field(default_factory=list)
    source_quality_notes: list[str] = Field(default_factory=list)
