from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class RecommendationAction(str, Enum):
    BUY_CANDIDATE = "buy_candidate"
    WATCH = "watch"
    AVOID = "avoid"
    HOLD = "hold"
    RISK_WARNING = "risk_warning"
    REDUCE = "reduce"
    TAKE_PROFIT = "take_profit"
    SELL_ALERT = "sell_alert"
    STOP_LOSS = "stop_loss"


class ActionLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Recommendation(BaseModel):
    symbol: str
    stock_name: str | None = None
    source_type: str = "unknown"
    as_of_time: datetime
    action: RecommendationAction
    action_level: ActionLevel
    confidence: float = Field(ge=0, le=1)
    suggested_horizon_days: list[int] = Field(default_factory=list)
    reason: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    invalid_conditions: list[str] = Field(default_factory=list)
    ai_view: str | None = None
    quant_decision: str | None = None
    final_level: str | None = None
    final_score: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
