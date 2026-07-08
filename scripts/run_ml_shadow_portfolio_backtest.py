from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.ml.ml_preprocess import MLPreprocessor


DAILY_COLUMNS = [
    "as_of_date", "model_train_end", "sample_count", "top_k", "avg_probability",
    "gross_return", "net_return", "hit_rate", "turnover", "universe_return",
    "random_topk_return", "bottom_k_return", "excess_vs_universe",
    "excess_vs_random", "top_bottom_spread",
]

TRADE_COLUMNS = [
    "as_of_date", "symbol", "stock_name", "predicted_probability", "rank",
    "future_return_5d", "net_return", "future_excess_return_5d",
    "future_rank_pct_5d", "return_5d", "return_20d", "risk_score",
    "overheat_score", "breadth_up_ratio_5d", "breadth_above_ma20_ratio",
]


def _date(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d")


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if np.isfinite(result) else None


def _mean(values: list[Any] | pd.Series) -> float | None:
    numeric = [_number(value) for value in values]
    valid = [value for value in numeric if value is not None]
    return None if not valid else float(np.mean(valid))


def _median(values: list[Any] | pd.Series) -> float | None:
    numeric = [_number(value) for value in values]
    valid = [value for value in numeric if value is not None]
    return None if not valid else float(np.median(valid))


def _max_drawdown_like(daily_net_returns: pd.Series) -> float | None:
    returns = pd.to_numeric(daily_net_returns, errors="coerce").dropna()
    if returns.empty:
        return None
    equity = 1.0 + returns.cumsum().to_numpy(dtype=float)
    running_max = np.maximum.accumulate(np.concatenate(([1.0], equity)))[1:]
    drawdown = equity / running_max - 1.0
    return float(np.min(drawdown))


def _first_train_end(args: argparse.Namespace, backtest_start: pd.Timestamp) -> pd.Timestamp:
    configured = pd.Timestamp(args.initial_train_end)
    previous_day = backtest_start - pd.Timedelta(days=1)
    # The fallback keeps the documented short smoke command usable when its date
    # range is earlier than the production default initial-train-end.
    return configured if configured < backtest_start else previous_day


def _train_model(
    frame: pd.DataFrame,
    dates: pd.Series,
    target_values: pd.Series,
    target: str,
    train_end: pd.Timestamp,
    min_train_samples: int,
) -> tuple[Any, list[str]] | None:
    train = frame.loc[dates.le(train_end) & target_values.notna()].copy()
    if len(train) < min_train_samples:
        return None
    preprocessor = MLPreprocessor()
    features = preprocessor.select_features(train, target)
    y_train = pd.to_numeric(train[target], errors="coerce").astype(int)
    if not features or y_train.nunique() < 2:
        return None
    pipeline = preprocessor.build_pipeline("logistic")
    try:
        pipeline.fit(preprocessor.numeric_frame(train, features), y_train)
    except Exception:
        return None
    return pipeline, features


def _random_topk_return(
    values: np.ndarray,
    top_k: int,
    rng: np.random.Generator,
    repeats: int = 100,
) -> float:
    simulations = [
        float(np.mean(rng.choice(values, size=top_k, replace=False)))
        for _ in range(repeats)
    ]
    return float(np.mean(simulations))


def _trade_rows(top: pd.DataFrame, cost_rate: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rank, (_, item) in enumerate(top.iterrows(), start=1):
        future_return = _number(item.get("future_return_5d"))
        rows.append({
            "as_of_date": str(item.get("as_of_date")),
            "symbol": item.get("symbol"),
            "stock_name": item.get("stock_name"),
            "predicted_probability": _number(item.get("predicted_probability")),
            "rank": rank,
            "future_return_5d": future_return,
            "net_return": None if future_return is None else future_return - cost_rate,
            "future_excess_return_5d": _number(item.get("future_excess_return_5d")),
            "future_rank_pct_5d": _number(item.get("future_rank_pct_5d")),
            "return_5d": _number(item.get("return_5d")),
            "return_20d": _number(item.get("return_20d")),
            "risk_score": _number(item.get("risk_score")),
            "overheat_score": _number(item.get("overheat_score")),
            "breadth_up_ratio_5d": _number(item.get("breadth_up_ratio_5d")),
            "breadth_above_ma20_ratio": _number(item.get("breadth_above_ma20_ratio")),
        })
    return rows


def _daily_result(
    group: pd.DataFrame,
    model_train_end: pd.Timestamp,
    top_k: int,
    cost_rate: float,
    previous_symbols: set[str] | None,
    rng: np.random.Generator,
) -> tuple[dict[str, Any], list[dict[str, Any]], set[str]]:
    ordered = group.sort_values("predicted_probability", ascending=False, kind="stable")
    top = ordered.head(top_k).copy()
    bottom = ordered.tail(top_k).copy()
    top_returns = pd.to_numeric(top["future_return_5d"], errors="coerce")
    universe_returns = pd.to_numeric(group["future_return_5d"], errors="coerce").dropna()
    bottom_returns = pd.to_numeric(bottom["future_return_5d"], errors="coerce")
    gross_return = _mean(top_returns)
    net_return = None if gross_return is None else gross_return - cost_rate
    universe_return = _mean(universe_returns)
    random_return = _random_topk_return(
        universe_returns.to_numpy(dtype=float), top_k, rng
    )
    bottom_return = _mean(bottom_returns)
    current_symbols = set(top["symbol"].astype(str))
    turnover = (
        1.0
        if previous_symbols is None
        else 1.0 - len(previous_symbols.intersection(current_symbols)) / top_k
    )
    as_of_date = str(group["as_of_date"].iloc[0])
    row = {
        "as_of_date": as_of_date,
        "model_train_end": model_train_end.date().isoformat(),
        "sample_count": int(len(group)),
        "top_k": top_k,
        "avg_probability": _mean(top["predicted_probability"]),
        "gross_return": gross_return,
        "net_return": net_return,
        "hit_rate": float(top_returns.dropna().gt(0).mean()),
        "turnover": float(turnover),
        "universe_return": universe_return,
        "random_topk_return": random_return,
        "bottom_k_return": bottom_return,
        "excess_vs_universe": (
            None if net_return is None or universe_return is None
            else net_return - universe_return
        ),
        "excess_vs_random": (
            None if net_return is None else net_return - random_return
        ),
        "top_bottom_spread": (
            None if gross_return is None or bottom_return is None
            else gross_return - bottom_return
        ),
    }
    return row, _trade_rows(top, cost_rate), current_symbols


def _summary(args: argparse.Namespace, daily_rows: list[dict[str, Any]]) -> dict[str, Any]:
    daily = pd.DataFrame(daily_rows, columns=DAILY_COLUMNS)
    avg_net = _mean(daily["net_return"]) if not daily.empty else None
    avg_universe = _mean(daily["universe_return"]) if not daily.empty else None
    avg_random = _mean(daily["random_topk_return"]) if not daily.empty else None
    avg_spread = _mean(daily["top_bottom_spread"]) if not daily.empty else None
    positive_excess = (
        0.0
        if daily.empty
        else float(pd.to_numeric(daily["excess_vs_universe"], errors="coerce").gt(0).mean())
    )
    if (
        avg_net is not None and avg_universe is not None and avg_random is not None
        and avg_spread is not None and avg_net > avg_universe and avg_net > avg_random
        and avg_spread > 0 and positive_excess >= 0.55
    ):
        status = "promising"
    elif avg_net is not None and avg_universe is not None and avg_net > avg_universe:
        status = "weak"
    else:
        status = "failed"
    return {
        "experiment_name": args.experiment_name,
        "dataset": args.dataset,
        "target": args.target,
        "model": args.model,
        "top_k": args.top_k,
        "holding_days": args.holding_days,
        "transaction_cost_bps": args.transaction_cost_bps,
        "slippage_bps": args.slippage_bps,
        "backtest_start": args.backtest_start.date().isoformat(),
        "backtest_end": args.backtest_end.date().isoformat(),
        "trading_days": len(daily),
        "avg_gross_return": _mean(daily["gross_return"]) if not daily.empty else None,
        "avg_net_return": avg_net,
        "median_net_return": _median(daily["net_return"]) if not daily.empty else None,
        "net_hit_day_rate": (
            0.0 if daily.empty else float(
                pd.to_numeric(daily["net_return"], errors="coerce").gt(0).mean()
            )
        ),
        "avg_hit_rate": _mean(daily["hit_rate"]) if not daily.empty else None,
        "avg_turnover": _mean(daily["turnover"]) if not daily.empty else None,
        "avg_universe_return": avg_universe,
        "avg_random_topk_return": avg_random,
        "avg_bottom_k_return": _mean(daily["bottom_k_return"]) if not daily.empty else None,
        "avg_excess_vs_universe": (
            _mean(daily["excess_vs_universe"]) if not daily.empty else None
        ),
        "avg_excess_vs_random": (
            _mean(daily["excess_vs_random"]) if not daily.empty else None
        ),
        "avg_top_bottom_spread": avg_spread,
        "max_drawdown_like": (
            _max_drawdown_like(daily["net_return"]) if not daily.empty else None
        ),
        "positive_excess_day_rate": positive_excess,
        "status": status,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Research-only ML shadow portfolio backtest")
    parser.add_argument(
        "--dataset",
        default="data/ml/ml_research_dataset_liquid500_context_relative_daily_2025_2026.csv",
    )
    parser.add_argument("--target", default="future_top30_5d")
    parser.add_argument("--model", default="logistic", choices=["logistic"])
    parser.add_argument("--experiment-name", default="ml_shadow_portfolio_top20_5d")
    parser.add_argument("--initial-train-end", type=_date, default=_date("2025-09-30"))
    parser.add_argument("--backtest-start", type=_date, default=_date("2025-10-01"))
    parser.add_argument("--backtest-end", type=_date, default=_date("2026-06-20"))
    parser.add_argument("--rebalance-frequency", default="daily", choices=["daily"])
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--holding-days", type=int, default=5)
    parser.add_argument("--transaction-cost-bps", type=float, default=10)
    parser.add_argument("--slippage-bps", type=float, default=10)
    parser.add_argument("--min-train-samples", type=int, default=5000)
    parser.add_argument("--min-daily-samples", type=int, default=30)
    parser.add_argument(
        "--retrain-monthly", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--output-dir", default="data/ml/shadow_backtest")
    args = parser.parse_args()

    if args.backtest_start > args.backtest_end:
        parser.error("backtest-start must not be later than backtest-end")
    if args.top_k <= 0 or args.min_train_samples <= 0 or args.min_daily_samples <= 0:
        parser.error("top-k and minimum sample counts must be positive")
    if args.holding_days != 5:
        parser.error("v1.0 currently requires --holding-days 5")
    if args.transaction_cost_bps < 0 or args.slippage_bps < 0:
        parser.error("cost and slippage must not be negative")
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        parser.error(f"dataset not found: {dataset_path}")
    frame = pd.read_csv(dataset_path, encoding="utf-8-sig")
    required = {"as_of_date", "symbol", args.target, "future_return_5d"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        parser.error(f"dataset missing columns: {missing}")

    dates = pd.to_datetime(frame["as_of_date"], errors="coerce")
    target_values = pd.to_numeric(frame[args.target], errors="coerce")
    backtest_start = pd.Timestamp(args.backtest_start)
    backtest_end = pd.Timestamp(args.backtest_end)
    eligible_dates = sorted(
        date for date in dates.loc[
            dates.ge(backtest_start) & dates.le(backtest_end) & target_values.notna()
        ].dropna().unique()
    )
    first_train_end = _first_train_end(args, backtest_start)
    models: dict[str, tuple[pd.Timestamp, Any, list[str]] | None] = {}
    daily_rows: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []
    previous_symbols: set[str] | None = None
    rng = np.random.default_rng(42)
    cost_rate = 2.0 * (args.transaction_cost_bps + args.slippage_bps) / 10000.0

    for raw_date in eligible_dates:
        as_of = pd.Timestamp(raw_date)
        month_key = as_of.strftime("%Y-%m") if args.retrain_monthly else "fixed"
        if month_key not in models:
            if not args.retrain_monthly or not models:
                train_end = first_train_end
            else:
                train_end = as_of.replace(day=1) - pd.Timedelta(days=1)
            trained = _train_model(
                frame, dates, target_values, args.target, train_end, args.min_train_samples
            )
            models[month_key] = (
                None if trained is None else (train_end, trained[0], trained[1])
            )
        model_package = models[month_key]
        if model_package is None:
            continue
        train_end, pipeline, features = model_package
        group = frame.loc[dates.eq(as_of) & target_values.notna()].copy()
        group = group.loc[pd.to_numeric(group["future_return_5d"], errors="coerce").notna()]
        if len(group) < max(args.min_daily_samples, args.top_k * 2):
            continue
        try:
            group["predicted_probability"] = pipeline.predict_proba(
                MLPreprocessor.numeric_frame(group, features)
            )[:, 1]
        except Exception:
            continue
        daily, selected_trades, previous_symbols = _daily_result(
            group, train_end, args.top_k, cost_rate, previous_symbols, rng
        )
        daily_rows.append(daily)
        trades.extend(selected_trades)

    summary = _summary(args, daily_rows)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_daily = output_dir / f"{args.experiment_name}_daily.csv"
    output_trades = output_dir / f"{args.experiment_name}_trades.csv"
    output_json = output_dir / f"{args.experiment_name}_summary.json"
    pd.DataFrame(daily_rows, columns=DAILY_COLUMNS).to_csv(
        output_daily, index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(trades, columns=TRADE_COLUMNS).to_csv(
        output_trades, index=False, encoding="utf-8-sig"
    )
    output_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )

    print("StockLens ML Shadow Portfolio Backtest")
    print(f"experiment: {args.experiment_name}")
    print(f"top_k: {args.top_k}")
    print(f"holding_days: {args.holding_days}")
    print(f"trading_days: {summary['trading_days']}")
    print(f"avg_gross_return: {summary['avg_gross_return']}")
    print(f"avg_net_return: {summary['avg_net_return']}")
    print(f"avg_universe_return: {summary['avg_universe_return']}")
    print(f"avg_random_topk_return: {summary['avg_random_topk_return']}")
    print(f"avg_excess_vs_universe: {summary['avg_excess_vs_universe']}")
    print(f"avg_excess_vs_random: {summary['avg_excess_vs_random']}")
    print(f"avg_top_bottom_spread: {summary['avg_top_bottom_spread']}")
    print(f"avg_turnover: {summary['avg_turnover']}")
    print(f"max_drawdown_like: {summary['max_drawdown_like']}")
    print(f"positive_excess_day_rate: {summary['positive_excess_day_rate']}")
    print(f"status: {summary['status']}")
    print(f"output_daily: {output_daily}")
    print(f"output_trades: {output_trades}")
    print(f"output_json: {output_json}")


if __name__ == "__main__":
    main()
