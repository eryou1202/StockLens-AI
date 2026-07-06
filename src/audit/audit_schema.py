from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, model_validator


class AuditMode(str, Enum):
    QUANT_ONLY_AUDIT = "quant_only_audit"


class AuditRequest(BaseModel):
    start_date: datetime
    end_date: datetime
    symbols: list[str]
    step_days: int = Field(default=5, ge=1)
    lookback_days: int = Field(default=120, ge=30)
    horizons: list[int] = Field(default_factory=lambda: [1, 3, 5, 10])
    max_symbols: int | None = Field(default=None, ge=1)
    audit_name: str | None = None
    mode: AuditMode = AuditMode.QUANT_ONLY_AUDIT

    @model_validator(mode="after")
    def validate_request(self) -> "AuditRequest":
        if self.start_date > self.end_date:
            raise ValueError("start_date 不能晚于 end_date")
        if not self.symbols:
            raise ValueError("symbols 不能为空")
        if any(value not in {1, 3, 5, 10} for value in self.horizons):
            raise ValueError("第一版 horizons 仅支持 1/3/5/10")
        return self


class AuditSample(BaseModel):
    audit_id: int
    symbol: str
    stock_name: str | None = None
    as_of_time: datetime
    mode: AuditMode = AuditMode.QUANT_ONLY_AUDIT
    action: str | None = None
    action_level: str | None = None
    final_score: float | None = None
    quant_score: float | None = None
    quant_decision: str | None = None
    ai_view: str | None = None
    current_price: float | None = None
    future_return_1d: float | None = None
    future_return_3d: float | None = None
    future_return_5d: float | None = None
    future_return_10d: float | None = None
    future_max_drawdown_5d: float | None = None
    future_max_drawdown_10d: float | None = None
    trend_score: float | None = None
    momentum_score: float | None = None
    volume_score: float | None = None
    risk_score: float | None = None
    overheat_score: float | None = None
    macd_score: float | None = None
    return_5d: float | None = None
    return_20d: float | None = None
    close_ma20_gap: float | None = None
    ma5_ma20_gap: float | None = None
    ma20_ma60_gap: float | None = None
    rsi_14: float | None = None
    macd_hist: float | None = None
    volume_ratio_5d: float | None = None
    max_drawdown_20d: float | None = None
    volatility_20d: float | None = None
    sample_note: str | None = None
    error_message: str | None = None

    @property
    def is_complete(self) -> bool:
        return (
            self.error_message is None
            and self.future_return_5d is not None
            and self.future_return_10d is not None
        )


class AuditSummary(BaseModel):
    audit_id: int
    audit_name: str | None = None
    start_date: datetime
    end_date: datetime
    symbols_count: int
    samples_count: int
    complete_samples: int
    action_distribution: dict[str, int] = Field(default_factory=dict)
    action_metrics: dict[str, dict[str, Any]] = Field(default_factory=dict)
    quant_decision_metrics: dict[str, dict[str, Any]] = Field(default_factory=dict)
    score_future_return_corr_5d: float | None = None
    score_future_return_corr_10d: float | None = None
    final_score_future_return_corr_5d: float | None = None
    final_score_future_return_corr_10d: float | None = None
    ranking_warning: bool = False
    metrics: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
