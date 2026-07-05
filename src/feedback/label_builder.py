from __future__ import annotations

import math
from datetime import datetime, time
from typing import Any

from src.feedback.feedback_schema import FutureLabel
from src.models.schemas import MarketBar, MarketDataBundle


class FutureLabelBuilder:
    """按交易日位置对齐信号，并计算未来收益与回撤标签。"""

    HORIZONS = (1, 3, 5, 10)

    def build(
        self,
        symbol: str,
        stock_name: str | None,
        as_of_time: datetime,
        market_data: MarketDataBundle,
    ) -> FutureLabel:
        bars = sorted(market_data.bars, key=lambda bar: bar.trade_time)
        entry_index = self._find_entry_index(bars, as_of_time)
        entry_rule = self._entry_rule(bars, as_of_time)
        base = {
            "symbol": symbol,
            "stock_name": stock_name,
            "as_of_time": as_of_time,
        }
        base_metadata = {
            "bar_count": len(bars),
            "horizons": list(self.HORIZONS),
            "entry_rule": entry_rule,
        }

        if entry_index is None:
            return FutureLabel(
                **base,
                feedback_status="failed",
                error_message="entry_bar_not_found",
                metadata=base_metadata,
            )

        entry_bar = bars[entry_index]
        entry_close = self._positive_float(entry_bar.close)
        if entry_close is None:
            return FutureLabel(
                **base,
                entry_trade_date=entry_bar.trade_time,
                feedback_status="failed",
                error_message="entry_close_invalid",
                metadata=base_metadata,
            )

        dates: dict[int, datetime | None] = {}
        closes: dict[int, float | None] = {}
        returns: dict[int, float | None] = {}
        hits: dict[int, bool | None] = {}
        for horizon in self.HORIZONS:
            future_index = entry_index + horizon
            future_bar = bars[future_index] if future_index < len(bars) else None
            dates[horizon] = future_bar.trade_time if future_bar else None
            closes[horizon] = self._positive_float(future_bar.close) if future_bar else None
            returns[horizon] = (
                closes[horizon] / entry_close - 1.0
                if closes[horizon] is not None
                else None
            )
            hits[horizon] = returns[horizon] > 0 if returns[horizon] is not None else None

        available = sum(value is not None for value in returns.values())
        if available == len(self.HORIZONS):
            status = "complete"
        elif available > 0:
            status = "partial"
        else:
            status = "pending"

        return FutureLabel(
            **base,
            entry_trade_date=entry_bar.trade_time,
            entry_close=entry_close,
            future_trade_date_1d=dates[1],
            future_trade_date_3d=dates[3],
            future_trade_date_5d=dates[5],
            future_trade_date_10d=dates[10],
            future_close_1d=closes[1],
            future_close_3d=closes[3],
            future_close_5d=closes[5],
            future_close_10d=closes[10],
            future_return_1d=returns[1],
            future_return_3d=returns[3],
            future_return_5d=returns[5],
            future_return_10d=returns[10],
            future_max_drawdown_5d=self._future_drawdown(bars, entry_index, entry_close, 5),
            future_max_drawdown_10d=self._future_drawdown(bars, entry_index, entry_close, 10),
            hit_1d=hits[1],
            hit_3d=hits[3],
            hit_5d=hits[5],
            hit_10d=hits[10],
            feedback_status=status,
            metadata={
                **base_metadata,
                "entry_index": entry_index,
                "available_horizons": available,
            },
        )

    @classmethod
    def _find_entry_index(cls, bars: list[MarketBar], as_of_time: datetime) -> int | None:
        index, _ = cls._select_entry(bars, as_of_time)
        return index

    @classmethod
    def _entry_rule(cls, bars: list[MarketBar], as_of_time: datetime) -> str:
        _, rule = cls._select_entry(bars, as_of_time)
        return rule

    @staticmethod
    def _select_entry(
        bars: list[MarketBar],
        as_of_time: datetime,
    ) -> tuple[int | None, str]:
        signal_date = as_of_time.date()
        same_day = [index for index, bar in enumerate(bars) if bar.trade_time.date() == signal_date]
        previous = [index for index, bar in enumerate(bars) if bar.trade_time.date() < signal_date]

        signal_clock = as_of_time.time().replace(tzinfo=None)
        if same_day and signal_clock >= time(15, 0):
            return same_day[-1], "after_close_same_day"
        if same_day:
            return (previous[-1] if previous else None), "intraday_previous_trade_day"
        return (previous[-1] if previous else None), "non_trading_day_previous_trade_day"

    @classmethod
    def _future_drawdown(
        cls,
        bars: list[MarketBar],
        entry_index: int,
        entry_close: float,
        horizon: int,
    ) -> float | None:
        future = bars[entry_index + 1 : entry_index + horizon + 1]
        if len(future) < horizon:
            return None
        prices: list[float] = []
        for bar in future:
            price = cls._positive_float(bar.low)
            if price is None:
                price = cls._positive_float(bar.close)
            if price is None:
                return None
            prices.append(price)
        return min(0.0, min(prices) / entry_close - 1.0)

    @staticmethod
    def _positive_float(value: Any) -> float | None:
        if value is None:
            return None
        try:
            result = float(value)
        except (TypeError, ValueError, OverflowError):
            return None
        return result if math.isfinite(result) and result > 0 else None
