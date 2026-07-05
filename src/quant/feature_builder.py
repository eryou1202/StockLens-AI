from __future__ import annotations

import math
import statistics
from typing import Any

from src.models.schemas import MarketBar, MarketDataBundle
from src.quant.feature_schema import QuantFeatureVector


class QuantFeatureBuilder:
    """将标准行情数据转换为可复用的量化特征。"""

    def build(self, market_data: MarketDataBundle) -> QuantFeatureVector:
        bars = sorted(market_data.bars, key=lambda bar: bar.trade_time)
        latest = bars[-1] if bars else None
        closes = [value for bar in bars if (value := self._valid_close(bar.close)) is not None]

        latest_close = self._valid_close(latest.close) if latest else None
        data_rows = len(closes)
        data_quality_score = self._data_quality_score(data_rows)
        warnings: list[str] = []

        if latest is not None and latest_close is None:
            data_quality_score = min(data_quality_score, 30.0)
            warnings.append("latest_close_invalid")

        is_latest_suspended = self._is_suspended(latest.trade_status) if latest else False
        if is_latest_suspended:
            warnings.append("latest_suspended")

        is_st = self._safe_bool(latest.is_st) if latest else None
        if is_st is True:
            warnings.append("is_st_stock")

        is_data_sufficient = data_rows >= 30 and latest_close is not None
        if not is_data_sufficient:
            warnings.append("insufficient_data")

        returns = {days: self._return(closes, days) for days in (1, 3, 5, 10, 20, 60)}
        moving_averages = {window: self._moving_average(closes, window) for window in (5, 10, 20, 60)}

        volatility_20d = self._volatility(closes, 20)
        max_drawdown_20d = self._max_drawdown(closes, 20)
        atr_14 = self._atr(bars, latest_close, 14)
        rsi_14 = self._rsi(closes, 14)
        macd, macd_signal, macd_hist = self._macd(closes)
        if len(closes) < 35:
            warnings.append("macd_warmup_short")
        bollinger_position, bollinger_percent_b = self._bollinger(closes, 20)

        return QuantFeatureVector(
            symbol=market_data.symbol,
            provider=market_data.provider,
            frequency=market_data.frequency,
            adjust_type=market_data.adjust_type,
            data_rows=data_rows,
            data_quality_score=data_quality_score,
            is_data_sufficient=is_data_sufficient,
            is_latest_suspended=is_latest_suspended,
            is_st=is_st,
            latest_open=self._safe_float(latest.open) if latest else None,
            latest_high=self._safe_float(latest.high) if latest else None,
            latest_low=self._safe_float(latest.low) if latest else None,
            latest_close=latest_close,
            latest_pre_close=self._safe_float(latest.pre_close) if latest else None,
            latest_volume=self._safe_float(latest.volume) if latest else None,
            latest_amount=self._safe_float(latest.amount) if latest else None,
            latest_turnover_rate=self._safe_float(latest.turnover_rate) if latest else None,
            return_1d=returns[1],
            return_3d=returns[3],
            return_5d=returns[5],
            return_10d=returns[10],
            return_20d=returns[20],
            return_60d=returns[60],
            ma5=moving_averages[5],
            ma10=moving_averages[10],
            ma20=moving_averages[20],
            ma60=moving_averages[60],
            ma5_ma20_gap=self._relative_gap(moving_averages[5], moving_averages[20]),
            ma10_ma20_gap=self._relative_gap(moving_averages[10], moving_averages[20]),
            ma20_ma60_gap=self._relative_gap(moving_averages[20], moving_averages[60]),
            close_ma20_gap=self._relative_gap(latest_close, moving_averages[20]),
            volume_ratio_5d=self._latest_ratio(bars, "volume", 5),
            volume_ratio_20d=self._latest_ratio(bars, "volume", 20),
            amount_ratio_5d=self._latest_ratio(bars, "amount", 5),
            amount_ratio_20d=self._latest_ratio(bars, "amount", 20),
            volatility_20d=volatility_20d,
            max_drawdown_20d=max_drawdown_20d,
            atr_14=atr_14,
            rsi_14=rsi_14,
            macd=macd,
            macd_signal=macd_signal,
            macd_hist=macd_hist,
            bollinger_position=bollinger_position,
            bollinger_percent_b=bollinger_percent_b,
            warnings=list(dict.fromkeys(warnings)),
            metadata={
                "raw_rows": len(bars),
                "valid_close_rows": data_rows,
                "latest_trade_time": latest.trade_time.isoformat() if latest else None,
                "source_data_quality": market_data.data_quality,
            },
        )

    @staticmethod
    def _safe_float(value: Any) -> float | None:
        if value is None:
            return None
        try:
            result = float(value)
        except (TypeError, ValueError, OverflowError):
            return None
        return result if math.isfinite(result) else None

    @classmethod
    def _valid_close(cls, value: Any) -> float | None:
        result = cls._safe_float(value)
        return result if result is not None and result > 0 else None

    @staticmethod
    def _safe_bool(value: Any) -> bool | None:
        if value is None:
            return None
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            if float(value) == 1.0:
                return True
            if float(value) == 0.0:
                return False
            return None
        normalized = str(value).strip().lower()
        if normalized in {"1", "true", "yes", "y", "t"}:
            return True
        if normalized in {"0", "false", "no", "n", "f"}:
            return False
        return None

    @staticmethod
    def _is_suspended(trade_status: Any) -> bool:
        return trade_status is not None and str(trade_status).strip() != "1"

    @staticmethod
    def _data_quality_score(data_rows: int) -> float:
        if data_rows >= 80:
            return 100.0
        if data_rows >= 60:
            return 85.0
        if data_rows >= 30:
            return 60.0
        if data_rows >= 20:
            return 40.0
        return 20.0

    @staticmethod
    def _return(closes: list[float], days: int) -> float | None:
        if len(closes) < days + 1:
            return None
        old = closes[-days - 1]
        return None if old <= 0 else closes[-1] / old - 1.0

    @staticmethod
    def _moving_average(values: list[float], window: int) -> float | None:
        if len(values) < window:
            return None
        return sum(values[-window:]) / window

    @staticmethod
    def _relative_gap(numerator: float | None, denominator: float | None) -> float | None:
        if numerator is None or denominator is None or denominator == 0:
            return None
        return numerator / denominator - 1.0

    @classmethod
    def _latest_ratio(cls, bars: list[MarketBar], field: str, window: int) -> float | None:
        if len(bars) < window + 1:
            return None
        current = cls._safe_float(getattr(bars[-1], field, None))
        history = [cls._safe_float(getattr(bar, field, None)) for bar in bars[-window - 1 : -1]]
        if current is None or current < 0 or any(value is None or value < 0 for value in history):
            return None
        average = sum(value for value in history if value is not None) / window
        return None if average == 0 else current / average

    @staticmethod
    def _volatility(closes: list[float], window: int) -> float | None:
        if len(closes) < window + 1:
            return None
        data = closes[-window - 1 :]
        returns = [data[index] / data[index - 1] - 1.0 for index in range(1, len(data))]
        return statistics.pstdev(returns)

    @staticmethod
    def _max_drawdown(closes: list[float], window: int) -> float | None:
        if len(closes) < window:
            return None
        running_max = closes[-window]
        worst = 0.0
        for close in closes[-window:]:
            running_max = max(running_max, close)
            worst = min(worst, close / running_max - 1.0)
        return worst

    @classmethod
    def _atr(cls, bars: list[MarketBar], latest_close: float | None, window: int) -> float | None:
        if latest_close is None or len(bars) < window + 1:
            return None
        true_ranges: list[float] = []
        recent_bars = bars[-window - 1 :]
        for index in range(1, len(recent_bars)):
            previous_close = cls._valid_close(recent_bars[index - 1].close)
            current_close = cls._valid_close(recent_bars[index].close)
            high = cls._safe_float(recent_bars[index].high)
            low = cls._safe_float(recent_bars[index].low)
            if previous_close is None or current_close is None or high is None or low is None or high < low:
                return None
            true_ranges.append(max(high - low, abs(high - previous_close), abs(low - previous_close)))
        return (sum(true_ranges) / window) / latest_close

    @staticmethod
    def _rsi(closes: list[float], window: int) -> float | None:
        if len(closes) < window + 1:
            return None
        recent = closes[-window - 1 :]
        deltas = [recent[index] - recent[index - 1] for index in range(1, len(recent))]
        average_gain = sum(max(delta, 0.0) for delta in deltas) / window
        average_loss = sum(max(-delta, 0.0) for delta in deltas) / window
        if average_gain == 0 and average_loss == 0:
            return 50.0
        if average_loss == 0:
            return 100.0
        if average_gain == 0:
            return 0.0
        relative_strength = average_gain / average_loss
        return 100.0 - 100.0 / (1.0 + relative_strength)

    @classmethod
    def _macd(cls, closes: list[float]) -> tuple[float | None, float | None, float | None]:
        if len(closes) < 26:
            return None, None, None
        ema12 = cls._ema_series(closes, 12)
        ema26 = cls._ema_series(closes, 26)
        dif = [fast - slow for fast, slow in zip(ema12, ema26)]
        signal = cls._ema_series(dif, 9)
        return dif[-1], signal[-1], dif[-1] - signal[-1]

    @staticmethod
    def _ema_series(values: list[float], window: int) -> list[float]:
        alpha = 2.0 / (window + 1.0)
        ema = [values[0]]
        for value in values[1:]:
            ema.append(alpha * value + (1.0 - alpha) * ema[-1])
        return ema

    @staticmethod
    def _bollinger(closes: list[float], window: int) -> tuple[float | None, float | None]:
        if len(closes) < window:
            return None, None
        recent = closes[-window:]
        mean = sum(recent) / window
        standard_deviation = statistics.pstdev(recent)
        if standard_deviation == 0:
            return 0.0, 0.5
        latest = recent[-1]
        position = (latest - mean) / (2.0 * standard_deviation)
        percent_b = (latest - (mean - 2.0 * standard_deviation)) / (4.0 * standard_deviation)
        return position, percent_b
