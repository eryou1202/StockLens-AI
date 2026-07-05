from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class FutureLabel(BaseModel):
    symbol: str
    stock_name: str | None = None
    as_of_time: datetime

    entry_trade_date: datetime | None = None
    entry_close: float | None = None

    future_trade_date_1d: datetime | None = None
    future_trade_date_3d: datetime | None = None
    future_trade_date_5d: datetime | None = None
    future_trade_date_10d: datetime | None = None

    future_close_1d: float | None = None
    future_close_3d: float | None = None
    future_close_5d: float | None = None
    future_close_10d: float | None = None

    future_return_1d: float | None = None
    future_return_3d: float | None = None
    future_return_5d: float | None = None
    future_return_10d: float | None = None

    future_max_drawdown_5d: float | None = None
    future_max_drawdown_10d: float | None = None

    hit_1d: bool | None = None
    hit_3d: bool | None = None
    hit_5d: bool | None = None
    hit_10d: bool | None = None

    feedback_status: Literal["pending", "partial", "complete", "failed"]
    error_message: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
