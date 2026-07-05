from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class PositionStatus(str, Enum):
    OPEN = "open"
    CLOSED = "closed"
    WATCH_ONLY = "watch_only"


class Position(BaseModel):
    id: int | None = None
    symbol: str
    stock_name: str | None = None
    entry_date: datetime
    entry_price: float = Field(gt=0)
    position_size: float | None = Field(default=None, gt=0)
    entry_reason: str | None = None
    entry_signal_id: str | None = None
    entry_action: str | None = None
    stop_loss_price: float | None = Field(default=None, gt=0)
    take_profit_price: float | None = Field(default=None, gt=0)
    max_holding_days: int | None = Field(default=None, gt=0)
    status: PositionStatus = PositionStatus.OPEN
    exit_date: datetime | None = None
    exit_price: float | None = Field(default=None, gt=0)
    exit_reason: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
