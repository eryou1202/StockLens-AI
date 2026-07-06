from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class TrackingStatus(str, Enum):
    TRACKING = "tracking"
    COMPLETE = "complete"
    FAILED = "failed"


class ManualVerdict(str, Enum):
    CORRECT = "correct"
    TOO_EARLY = "too_early"
    TOO_LATE = "too_late"
    FALSE_POSITIVE = "false_positive"
    MISSED_OPPORTUNITY = "missed_opportunity"
    DIRECTION_WRONG = "direction_wrong"
    RISK_UNDERESTIMATED = "risk_underestimated"
    RISK_OVERESTIMATED = "risk_overestimated"
    NEEDS_REVIEW = "needs_review"


class TrackedRecommendation(BaseModel):
    id: int | None = None
    symbol: str
    stock_name: str | None = None
    as_of_time: datetime
    source_type: str | None = None
    action: str
    action_level: str | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    final_score: float | None = None
    ai_view: str | None = None
    quant_decision: str | None = None
    final_level: str | None = None
    current_price: float | None = None
    suggested_horizon_days: list[int] = Field(default_factory=list)
    reason: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    invalid_conditions: list[str] = Field(default_factory=list)
    future_return_1d: float | None = None
    future_return_3d: float | None = None
    future_return_5d: float | None = None
    future_return_10d: float | None = None
    future_max_drawdown_5d: float | None = None
    future_max_drawdown_10d: float | None = None
    tracking_status: TrackingStatus = TrackingStatus.TRACKING
    manual_verdict: ManualVerdict | None = None
    manual_notes: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
