from __future__ import annotations

import math
import statistics
from datetime import datetime, timedelta
from typing import Any

from src.data.market_data_provider import MarketDataProvider
from src.ml.ml_schema import CONTEXT_FEATURE_COLUMNS
from src.models.schemas import MarketBar, MarketDataBundle


class MLContextFeatureBuilder:
    """Build leakage-safe cross-sectional context from bars visible at as-of time."""

    INDEXES = {
        "hs300": "000300.SH",
        "zz500": "000905.SH",
        "cyb": "399006.SZ",
    }

    def __init__(
        self,
        provider: MarketDataProvider,
        frequency: str = "1d",
        adjust_type: str = "qfq",
        lookback_days: int = 120,
    ) -> None:
        self.provider = provider
        self.frequency = frequency
        self.adjust_type = adjust_type
        self.lookback_days = lookback_days
        self.symbol_bundles: dict[str, MarketDataBundle] = {}
        self.index_bundles: dict[str, MarketDataBundle | None] = {}
        self._context_cache: dict[str, dict[str, float | None]] = {}
        self._warnings: dict[str, list[str]] = {}

    def prepare(
        self,
        symbol_bundles: dict[str, MarketDataBundle],
        start_date: datetime,
        end_date: datetime,
    ) -> None:
        self.symbol_bundles = dict(symbol_bundles)
        self._context_cache.clear()
        self._warnings.clear()
        for alias, symbol in self.INDEXES.items():
            try:
                self.index_bundles[alias] = self.provider.get_bars(
                    symbol=symbol,
                    start_time=start_date - timedelta(days=self.lookback_days + 10),
                    end_time=end_date,
                    frequency=self.frequency,
                    adjust_type=self.adjust_type,
                )
            except Exception as exc:
                self.index_bundles[alias] = None
                self._warnings.setdefault("global", []).append(
                    f"index_{alias}_unavailable:{type(exc).__name__}"
                )

    def build(
        self,
        as_of_date: datetime,
        symbol: str,
        symbols: list[str],
        stock_features: dict[str, Any] | None = None,
    ) -> dict[str, float | None]:
        key = as_of_date.date().isoformat()
        if key not in self._context_cache:
            self._context_cache[key] = self._build_shared_context(as_of_date, symbols)
        result = dict(self._context_cache[key])
        stock_features = stock_features or self._snapshot(
            self.symbol_bundles.get(symbol), as_of_date
        )
        result.update(self._relative_features(stock_features, result))
        return {column: result.get(column) for column in CONTEXT_FEATURE_COLUMNS}

    def warnings_for(self, as_of_date: datetime) -> list[str]:
        key = as_of_date.date().isoformat()
        return list(dict.fromkeys(self._warnings.get("global", []) + self._warnings.get(key, [])))

    @staticmethod
    def empty_features() -> dict[str, None]:
        return {column: None for column in CONTEXT_FEATURE_COLUMNS}

    def _build_shared_context(
        self,
        as_of_date: datetime,
        symbols: list[str],
    ) -> dict[str, float | None]:
        result: dict[str, float | None] = self.empty_features()
        key = as_of_date.date().isoformat()
        for alias in self.INDEXES:
            snapshot = self._snapshot(self.index_bundles.get(alias), as_of_date)
            if not snapshot:
                self._warnings.setdefault(key, []).append(f"index_{alias}_missing_at_as_of")
                continue
            for metric in (
                "return_1d", "return_5d", "return_20d", "ma5_ma20_gap",
                "volatility_20d", "max_drawdown_20d",
            ):
                result[f"market_{alias}_{metric}"] = snapshot.get(metric)

        snapshots = [
            snapshot
            for symbol in symbols
            if (snapshot := self._snapshot(self.symbol_bundles.get(symbol), as_of_date))
        ]
        if not snapshots:
            self._warnings.setdefault(key, []).append("breadth_universe_empty")
            return result

        for horizon in (1, 3, 5):
            values = self._values(snapshots, f"return_{horizon}d")
            result[f"breadth_up_ratio_{horizon}d"] = self._ratio(
                [value > 0 for value in values]
            )
        above = [item["above_ma20"] for item in snapshots if item.get("above_ma20") is not None]
        result["breadth_above_ma20_ratio"] = self._ratio(above)
        for horizon in (1, 5, 20):
            values = self._values(snapshots, f"return_{horizon}d")
            result[f"breadth_median_return_{horizon}d"] = self._median(values)

        positive_volume = [
            item["return_5d"] > 0 and item["volume_ratio_5d"] > 1.0
            for item in snapshots
            if item.get("return_5d") is not None and item.get("volume_ratio_5d") is not None
        ]
        result["breadth_positive_volume_ratio_5d"] = self._ratio(positive_volume)

        for style, metric in (
            ("momentum", "return_20d"),
            ("volatility", "volatility_20d"),
            ("activity", "amount_ratio_20d"),
        ):
            high, low = self._style_returns(snapshots, metric)
            result[f"style_high_{style}_return_5d"] = high
            result[f"style_low_{style}_return_5d"] = low
            result[f"style_{style}_spread_5d"] = (
                None if high is None or low is None else high - low
            )
        return result

    @staticmethod
    def _relative_features(
        stock: dict[str, Any],
        context: dict[str, float | None],
    ) -> dict[str, float | None]:
        def difference(left: Any, right: Any) -> float | None:
            left_value = MLContextFeatureBuilder._number(left)
            right_value = MLContextFeatureBuilder._number(right)
            return None if left_value is None or right_value is None else left_value - right_value

        return {
            "relative_to_hs300_return_5d": difference(
                stock.get("return_5d"), context.get("market_hs300_return_5d")
            ),
            "relative_to_hs300_return_20d": difference(
                stock.get("return_20d"), context.get("market_hs300_return_20d")
            ),
            "relative_to_zz500_return_5d": difference(
                stock.get("return_5d"), context.get("market_zz500_return_5d")
            ),
            "relative_to_zz500_return_20d": difference(
                stock.get("return_20d"), context.get("market_zz500_return_20d")
            ),
            "relative_to_breadth_median_return_5d": difference(
                stock.get("return_5d"), context.get("breadth_median_return_5d")
            ),
            "relative_to_breadth_median_return_20d": difference(
                stock.get("return_20d"), context.get("breadth_median_return_20d")
            ),
        }

    def _snapshot(
        self,
        bundle: MarketDataBundle | None,
        as_of_date: datetime,
    ) -> dict[str, Any]:
        if bundle is None:
            return {}
        start = as_of_date - timedelta(days=self.lookback_days)
        bars = [
            bar for bar in bundle.sorted_bars()
            if start <= self._align(bar.trade_time, as_of_date) <= as_of_date
            and self._positive(bar.close) is not None
        ]
        closes = [self._positive(bar.close) for bar in bars]
        valid_closes = [value for value in closes if value is not None]
        if not valid_closes:
            return {}
        ma20 = self._mean(valid_closes[-20:]) if len(valid_closes) >= 20 else None
        ma5 = self._mean(valid_closes[-5:]) if len(valid_closes) >= 5 else None
        latest = valid_closes[-1]
        return {
            "return_1d": self._return(valid_closes, 1),
            "return_3d": self._return(valid_closes, 3),
            "return_5d": self._return(valid_closes, 5),
            "return_20d": self._return(valid_closes, 20),
            "ma5_ma20_gap": self._gap(ma5, ma20),
            "above_ma20": None if ma20 is None else latest > ma20,
            "volatility_20d": self._volatility(valid_closes, 20),
            "max_drawdown_20d": self._max_drawdown(valid_closes, 20),
            "volume_ratio_5d": self._latest_ratio(bars, "volume", 5),
            "amount_ratio_20d": self._latest_ratio(bars, "amount", 20),
        }

    @classmethod
    def _style_returns(
        cls,
        snapshots: list[dict[str, Any]],
        metric: str,
    ) -> tuple[float | None, float | None]:
        pairs = [
            (metric_value, return_value)
            for item in snapshots
            if (metric_value := cls._number(item.get(metric))) is not None
            and (return_value := cls._number(item.get("return_5d"))) is not None
        ]
        if len(pairs) < 2:
            return None, None
        pairs.sort(key=lambda item: item[0])
        size = max(1, int(len(pairs) * 0.30))
        low = cls._mean([item[1] for item in pairs[:size]])
        high = cls._mean([item[1] for item in pairs[-size:]])
        return high, low

    @classmethod
    def _latest_ratio(cls, bars: list[MarketBar], field: str, window: int) -> float | None:
        if len(bars) <= window:
            return None
        current = cls._number(getattr(bars[-1], field, None))
        history = [cls._number(getattr(bar, field, None)) for bar in bars[-window - 1 : -1]]
        if current is None or current < 0 or any(value is None or value < 0 for value in history):
            return None
        average = cls._mean([value for value in history if value is not None])
        return None if average in (None, 0) else current / average

    @staticmethod
    def _return(values: list[float], horizon: int) -> float | None:
        return None if len(values) <= horizon else values[-1] / values[-horizon - 1] - 1.0

    @staticmethod
    def _gap(left: float | None, right: float | None) -> float | None:
        return None if left is None or right in (None, 0) else left / right - 1.0

    @staticmethod
    def _volatility(values: list[float], window: int) -> float | None:
        if len(values) <= window:
            return None
        recent = values[-window - 1 :]
        returns = [recent[index] / recent[index - 1] - 1.0 for index in range(1, len(recent))]
        return statistics.pstdev(returns)

    @staticmethod
    def _max_drawdown(values: list[float], window: int) -> float | None:
        if len(values) < window:
            return None
        peak = values[-window]
        worst = 0.0
        for value in values[-window:]:
            peak = max(peak, value)
            worst = min(worst, value / peak - 1.0)
        return worst

    @staticmethod
    def _values(items: list[dict[str, Any]], field: str) -> list[float]:
        return [value for item in items if (value := MLContextFeatureBuilder._number(item.get(field))) is not None]

    @staticmethod
    def _ratio(values: list[bool]) -> float | None:
        return None if not values else sum(bool(value) for value in values) / len(values)

    @staticmethod
    def _mean(values: list[float]) -> float | None:
        return None if not values else sum(values) / len(values)

    @staticmethod
    def _median(values: list[float]) -> float | None:
        return None if not values else float(statistics.median(values))

    @staticmethod
    def _number(value: Any) -> float | None:
        try:
            result = float(value)
        except (TypeError, ValueError, OverflowError):
            return None
        return result if math.isfinite(result) else None

    @classmethod
    def _positive(cls, value: Any) -> float | None:
        result = cls._number(value)
        return result if result is not None and result > 0 else None

    @staticmethod
    def _align(value: datetime, reference: datetime) -> datetime:
        if reference.tzinfo is None:
            return value.replace(tzinfo=None)
        if value.tzinfo is None:
            return value.replace(tzinfo=reference.tzinfo)
        return value.astimezone(reference.tzinfo)
