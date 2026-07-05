from __future__ import annotations

import pandas as pd

from src.models.schemas import MarketDataBundle


def bundle_to_frame(bundle: MarketDataBundle) -> pd.DataFrame:
    rows = [bar.model_dump() for bar in bundle.sorted_bars()]
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["trade_time"] = pd.to_datetime(df["trade_time"])
    return df.sort_values("trade_time").reset_index(drop=True)


def simple_return(series: pd.Series, days: int) -> float | None:
    if len(series) <= days:
        return None
    old = float(series.iloc[-days - 1])
    new = float(series.iloc[-1])
    if old == 0:
        return None
    return new / old - 1


def moving_average(series: pd.Series, window: int) -> float | None:
    if len(series) < window:
        return None
    return float(series.tail(window).mean())


def volume_ratio(volume: pd.Series, window: int = 5) -> float | None:
    if len(volume) < window + 1:
        return None
    recent = float(volume.iloc[-1])
    avg = float(volume.iloc[-window - 1 : -1].mean())
    if avg == 0:
        return None
    return recent / avg


def volatility(series: pd.Series, window: int = 20) -> float | None:
    if len(series) < window + 1:
        return None
    returns = series.pct_change().dropna().tail(window)
    if returns.empty:
        return None
    return float(returns.std())


def max_drawdown(series: pd.Series, window: int = 20) -> float | None:
    if len(series) < window:
        return None
    data = series.tail(window).astype(float)
    rolling_max = data.cummax()
    drawdown = data / rolling_max - 1
    return float(drawdown.min())


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
