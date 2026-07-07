from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


STANDARD_RETURN_HORIZONS = (1, 2, 3, 5, 10, 20, 60)
STANDARD_HIT_HORIZONS = (1, 2, 3, 5, 10, 20)
STANDARD_DRAWDOWN_HORIZONS = (3, 5, 10, 20)

CONTEXT_FEATURE_COLUMNS = [
    f"market_{index}_{metric}"
    for index in ("hs300", "zz500", "cyb")
    for metric in (
        "return_1d", "return_5d", "return_20d", "ma5_ma20_gap",
        "volatility_20d", "max_drawdown_20d",
    )
] + [
    "breadth_up_ratio_1d",
    "breadth_up_ratio_3d",
    "breadth_up_ratio_5d",
    "breadth_above_ma20_ratio",
    "breadth_median_return_1d",
    "breadth_median_return_5d",
    "breadth_median_return_20d",
    "breadth_positive_volume_ratio_5d",
    "relative_to_hs300_return_5d",
    "relative_to_hs300_return_20d",
    "relative_to_zz500_return_5d",
    "relative_to_zz500_return_20d",
    "relative_to_breadth_median_return_5d",
    "relative_to_breadth_median_return_20d",
    "style_high_momentum_return_5d",
    "style_low_momentum_return_5d",
    "style_momentum_spread_5d",
    "style_high_volatility_return_5d",
    "style_low_volatility_return_5d",
    "style_volatility_spread_5d",
    "style_high_activity_return_5d",
    "style_low_activity_return_5d",
    "style_activity_spread_5d",
]


class MLDatasetRequest(BaseModel):
    start_date: datetime
    end_date: datetime
    symbols: list[str]
    sample_interval_days: int = Field(default=5, ge=1)
    lookback_days: int = Field(default=120, ge=30)
    horizons: list[int] = Field(default_factory=lambda: [1, 2, 3, 5, 10, 20])
    max_symbols: int | None = Field(default=None, ge=1)
    include_context_features: bool = False
    output_path: str = "data/ml/ml_research_dataset.csv"

    @field_validator("symbols")
    @classmethod
    def validate_symbols(cls, value: list[str]) -> list[str]:
        result = [str(item).strip() for item in value if str(item).strip()]
        if not result:
            raise ValueError("symbols must not be empty")
        return result

    @field_validator("horizons")
    @classmethod
    def validate_horizons(cls, value: list[int]) -> list[int]:
        result = sorted(set(int(item) for item in value))
        unsupported = [item for item in result if item not in STANDARD_RETURN_HORIZONS]
        if unsupported:
            raise ValueError(f"unsupported horizons: {unsupported}")
        return result

    @model_validator(mode="after")
    def validate_date_range(self) -> "MLDatasetRequest":
        if self.start_date > self.end_date:
            raise ValueError("start_date must not be later than end_date")
        return self


class MLResearchSample(BaseModel):
    sample_id: str
    symbol: str
    stock_name: str | None = None
    as_of_date: str
    price_time: str | None = None
    current_price: float | None = None
    source: str = "unknown"
    sample_interval_days: int
    lookback_days: int

    return_1d: float | None = None
    return_2d: float | None = None
    return_3d: float | None = None
    return_5d: float | None = None
    return_10d: float | None = None
    return_20d: float | None = None
    ma5_ma20_gap: float | None = None
    ma20_ma60_gap: float | None = None
    close_ma20_gap: float | None = None
    trend_score: float | None = None
    momentum_score: float | None = None
    volume_score: float | None = None
    risk_score: float | None = None
    overheat_score: float | None = None
    macd_score: float | None = None
    rsi_14: float | None = None
    macd_hist: float | None = None
    volume_ratio_1d: float | None = None
    volume_ratio_3d: float | None = None
    volume_ratio_5d: float | None = None
    volume_ratio_20d: float | None = None
    amount_ratio_5d: float | None = None
    amount_ratio_20d: float | None = None
    max_drawdown_5d: float | None = None
    max_drawdown_20d: float | None = None
    volatility_5d: float | None = None
    volatility_20d: float | None = None
    atr_14: float | None = None
    bollinger_position: float | None = None
    quant_score: float | None = None
    quant_decision_encoded: int | None = None

    market_hs300_return_1d: float | None = None
    market_hs300_return_5d: float | None = None
    market_hs300_return_20d: float | None = None
    market_hs300_ma5_ma20_gap: float | None = None
    market_hs300_volatility_20d: float | None = None
    market_hs300_max_drawdown_20d: float | None = None
    market_zz500_return_1d: float | None = None
    market_zz500_return_5d: float | None = None
    market_zz500_return_20d: float | None = None
    market_zz500_ma5_ma20_gap: float | None = None
    market_zz500_volatility_20d: float | None = None
    market_zz500_max_drawdown_20d: float | None = None
    market_cyb_return_1d: float | None = None
    market_cyb_return_5d: float | None = None
    market_cyb_return_20d: float | None = None
    market_cyb_ma5_ma20_gap: float | None = None
    market_cyb_volatility_20d: float | None = None
    market_cyb_max_drawdown_20d: float | None = None

    breadth_up_ratio_1d: float | None = None
    breadth_up_ratio_3d: float | None = None
    breadth_up_ratio_5d: float | None = None
    breadth_above_ma20_ratio: float | None = None
    breadth_median_return_1d: float | None = None
    breadth_median_return_5d: float | None = None
    breadth_median_return_20d: float | None = None
    breadth_positive_volume_ratio_5d: float | None = None

    relative_to_hs300_return_5d: float | None = None
    relative_to_hs300_return_20d: float | None = None
    relative_to_zz500_return_5d: float | None = None
    relative_to_zz500_return_20d: float | None = None
    relative_to_breadth_median_return_5d: float | None = None
    relative_to_breadth_median_return_20d: float | None = None

    style_high_momentum_return_5d: float | None = None
    style_low_momentum_return_5d: float | None = None
    style_momentum_spread_5d: float | None = None
    style_high_volatility_return_5d: float | None = None
    style_low_volatility_return_5d: float | None = None
    style_volatility_spread_5d: float | None = None
    style_high_activity_return_5d: float | None = None
    style_low_activity_return_5d: float | None = None
    style_activity_spread_5d: float | None = None

    future_return_1d: float | None = None
    future_return_2d: float | None = None
    future_return_3d: float | None = None
    future_return_5d: float | None = None
    future_return_10d: float | None = None
    future_return_20d: float | None = None
    future_return_60d: float | None = None
    hit_1d: int | None = None
    hit_2d: int | None = None
    hit_3d: int | None = None
    hit_5d: int | None = None
    hit_10d: int | None = None
    hit_20d: int | None = None
    future_max_drawdown_3d: float | None = None
    future_max_drawdown_5d: float | None = None
    future_max_drawdown_10d: float | None = None
    future_max_drawdown_20d: float | None = None
    label_status: Literal["complete", "incomplete", "failed"] = "incomplete"
    label_error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict, exclude=True)


class MLTrainRequest(BaseModel):
    dataset_path: str
    target: str
    model_type: Literal["logistic", "random_forest_regressor"]
    train_end: datetime
    valid_start: datetime
    valid_end: datetime
    model_name: str
    notes: str | None = None
