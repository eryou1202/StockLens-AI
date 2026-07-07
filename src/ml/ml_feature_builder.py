from __future__ import annotations

import math
import statistics
from typing import Any

from src.models.schemas import MarketBar, MarketDataBundle
from src.quant.feature_builder import QuantFeatureBuilder
from src.quant.rule_scorer import RuleScorer


class MLFeatureBuilder:
    """Build model features from a bundle that has already been cut at as-of time."""

    DECISION_ENCODING = {"reject": -1, "uncertain": 0, "support": 1}

    def __init__(self) -> None:
        self.quant_builder = QuantFeatureBuilder()
        self.rule_scorer = RuleScorer()

    def build(self, market_data: MarketDataBundle) -> dict[str, Any]:
        bars = sorted(market_data.bars, key=lambda item: item.trade_time)
        closes = [value for bar in bars if (value := self._positive(bar.close)) is not None]
        features = self.quant_builder.build(market_data)
        scores = self.rule_scorer.score(features)
        latest = bars[-1] if bars else None
        return {
            "price_time": latest.trade_time.isoformat() if latest else None,
            "current_price": features.latest_close,
            "source": market_data.provider,
            "return_1d": self._return(closes, 1),
            "return_2d": self._return(closes, 2),
            "return_3d": self._return(closes, 3),
            "return_5d": self._return(closes, 5),
            "return_10d": self._return(closes, 10),
            "return_20d": self._return(closes, 20),
            "ma5_ma20_gap": features.ma5_ma20_gap,
            "ma20_ma60_gap": features.ma20_ma60_gap,
            "close_ma20_gap": features.close_ma20_gap,
            "trend_score": scores.trend_score,
            "momentum_score": scores.momentum_score,
            "volume_score": scores.volume_score,
            "risk_score": scores.risk_score,
            "overheat_score": scores.overheat_score,
            "macd_score": scores.macd_score,
            "rsi_14": features.rsi_14,
            "macd_hist": features.macd_hist,
            "volume_ratio_1d": self._latest_ratio(bars, "volume", 1),
            "volume_ratio_3d": self._latest_ratio(bars, "volume", 3),
            "volume_ratio_5d": features.volume_ratio_5d,
            "volume_ratio_20d": features.volume_ratio_20d,
            "amount_ratio_5d": features.amount_ratio_5d,
            "amount_ratio_20d": features.amount_ratio_20d,
            "max_drawdown_5d": self._max_drawdown(closes, 5),
            "max_drawdown_20d": features.max_drawdown_20d,
            "volatility_5d": self._volatility(closes, 5),
            "volatility_20d": features.volatility_20d,
            "atr_14": features.atr_14,
            "bollinger_position": features.bollinger_position,
            "quant_score": scores.quant_score,
            "quant_decision_encoded": self.DECISION_ENCODING.get(scores.quant_decision),
        }

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
    def _return(closes: list[float], horizon: int) -> float | None:
        if len(closes) <= horizon:
            return None
        return closes[-1] / closes[-horizon - 1] - 1.0

    @classmethod
    def _latest_ratio(cls, bars: list[MarketBar], field: str, window: int) -> float | None:
        if len(bars) <= window:
            return None
        current = cls._number(getattr(bars[-1], field, None))
        history = [cls._number(getattr(bar, field, None)) for bar in bars[-window - 1 : -1]]
        if current is None or any(item is None or item < 0 for item in history):
            return None
        average = sum(item for item in history if item is not None) / window
        return None if average <= 0 else current / average

    @staticmethod
    def _max_drawdown(closes: list[float], window: int) -> float | None:
        if len(closes) < window:
            return None
        peak = closes[-window]
        worst = 0.0
        for close in closes[-window:]:
            peak = max(peak, close)
            worst = min(worst, close / peak - 1.0)
        return worst

    @staticmethod
    def _volatility(closes: list[float], window: int) -> float | None:
        if len(closes) <= window:
            return None
        values = closes[-window - 1 :]
        returns = [values[index] / values[index - 1] - 1.0 for index in range(1, len(values))]
        return statistics.pstdev(returns)
