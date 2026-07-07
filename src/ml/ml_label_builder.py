from __future__ import annotations

import math
from datetime import datetime
from typing import Any

from src.ml.ml_schema import (
    STANDARD_DRAWDOWN_HORIZONS,
    STANDARD_HIT_HORIZONS,
    STANDARD_RETURN_HORIZONS,
)
from src.models.schemas import MarketBar, MarketDataBundle


class MLLabelBuilder:
    """Create future-only labels in memory; it never touches feedback storage."""

    def build(
        self,
        market_data: MarketDataBundle,
        as_of_time: datetime,
        required_horizons: list[int],
    ) -> dict[str, Any]:
        bars = [bar for bar in market_data.sorted_bars() if self._positive(bar.close) is not None]
        entry_index = self._entry_index(bars, as_of_time)
        result: dict[str, Any] = {}
        for horizon in STANDARD_RETURN_HORIZONS:
            result[f"future_return_{horizon}d"] = None
        for horizon in STANDARD_HIT_HORIZONS:
            result[f"hit_{horizon}d"] = None
        for horizon in STANDARD_DRAWDOWN_HORIZONS:
            result[f"future_max_drawdown_{horizon}d"] = None

        if entry_index is None:
            result.update(label_status="failed", label_error="no valid entry bar at or before as_of_date")
            return result
        entry = self._positive(bars[entry_index].close)
        if entry is None:
            result.update(label_status="failed", label_error="entry close is invalid")
            return result

        available = len(bars) - entry_index - 1
        for horizon in STANDARD_RETURN_HORIZONS:
            if available >= horizon:
                future = self._positive(bars[entry_index + horizon].close)
                if future is not None:
                    value = future / entry - 1.0
                    result[f"future_return_{horizon}d"] = value
                    if horizon in STANDARD_HIT_HORIZONS:
                        result[f"hit_{horizon}d"] = int(value > 0)
        for horizon in STANDARD_DRAWDOWN_HORIZONS:
            if available >= horizon:
                future_closes = [
                    self._positive(bar.close)
                    for bar in bars[entry_index + 1 : entry_index + horizon + 1]
                ]
                if all(value is not None for value in future_closes):
                    result[f"future_max_drawdown_{horizon}d"] = min(
                        value / entry - 1.0 for value in future_closes if value is not None
                    )

        missing = [
            horizon for horizon in required_horizons
            if result.get(f"future_return_{horizon}d") is None
        ]
        result["label_status"] = "complete" if not missing else "incomplete"
        result["label_error"] = None if not missing else f"insufficient future bars for horizons: {missing}"
        return result

    @classmethod
    def _entry_index(cls, bars: list[MarketBar], as_of_time: datetime) -> int | None:
        candidates = [
            index for index, bar in enumerate(bars)
            if cls._align(bar.trade_time, as_of_time) <= as_of_time
        ]
        return candidates[-1] if candidates else None

    @staticmethod
    def _align(value: datetime, reference: datetime) -> datetime:
        if reference.tzinfo is None:
            return value.replace(tzinfo=None)
        if value.tzinfo is None:
            return value.replace(tzinfo=reference.tzinfo)
        return value.astimezone(reference.tzinfo)

    @staticmethod
    def _positive(value: Any) -> float | None:
        try:
            result = float(value)
        except (TypeError, ValueError, OverflowError):
            return None
        return result if math.isfinite(result) and result > 0 else None
