from __future__ import annotations


def calc_unrealized_return(current_price: float, entry_price: float) -> float:
    if entry_price <= 0:
        raise ValueError("entry_price 必须大于 0")
    return current_price / entry_price - 1.0


def should_stop_loss(unrealized_return: float, stop_loss_threshold: float = -0.05) -> bool:
    return unrealized_return <= stop_loss_threshold


def should_take_profit(unrealized_return: float, overheat_score: float | None) -> bool:
    return unrealized_return >= 0.12 and overheat_score is not None and overheat_score >= 70


def should_reduce(
    unrealized_return: float,
    rsi_14: float | None,
    overheat_score: float | None,
) -> bool:
    return (
        unrealized_return >= 0.08
        and rsi_14 is not None
        and rsi_14 >= 75
        and overheat_score is not None
        and overheat_score >= 65
    )


def is_trend_broken(
    close_ma20_gap: float | None,
    macd_hist: float | None,
    momentum_score: float | None,
) -> bool:
    return (
        close_ma20_gap is not None
        and close_ma20_gap < 0
        and macd_hist is not None
        and macd_hist < 0
        and momentum_score is not None
        and momentum_score < 40
    )
