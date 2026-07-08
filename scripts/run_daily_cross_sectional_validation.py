from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.ml.ml_preprocess import MLPreprocessor


DAILY_COLUMNS = [
    "window_id", "as_of_date", "sample_count", "daily_spearman",
    "daily_top_avg_return", "daily_middle_avg_return", "daily_bottom_avg_return",
    "daily_top_bottom_spread", "daily_top_avg_excess_return",
    "daily_bottom_avg_excess_return", "daily_top_bottom_excess_spread",
    "daily_top_hit_rate", "daily_bottom_hit_rate",
]

WINDOW_COLUMNS = [
    "window_id", "train_end", "valid_start", "valid_end", "valid_days",
    "valid_samples", "avg_daily_spearman", "median_daily_spearman",
    "avg_daily_top_return", "avg_daily_middle_return", "avg_daily_bottom_return",
    "avg_daily_top_bottom_spread", "median_daily_top_bottom_spread",
    "avg_daily_top_excess_return", "avg_daily_bottom_excess_return",
    "avg_daily_top_bottom_excess_spread", "avg_daily_top_hit_rate",
    "avg_daily_bottom_hit_rate", "positive_day_rate", "status",
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


def _correlation(left: pd.Series, right: pd.Series) -> float | None:
    x = pd.to_numeric(left, errors="coerce")
    y = pd.to_numeric(right, errors="coerce")
    valid = x.notna() & y.notna()
    if int(valid.sum()) < 2 or x.loc[valid].nunique() < 2 or y.loc[valid].nunique() < 2:
        return None
    value = x.loc[valid].corr(y.loc[valid], method="spearman")
    return None if pd.isna(value) else float(value)


def _frame_mean(frame: pd.DataFrame, column: str) -> float | None:
    if column not in frame.columns:
        return None
    return _mean(pd.to_numeric(frame[column], errors="coerce"))


def _hit_rate(frame: pd.DataFrame, return_column: str) -> float | None:
    if return_column not in frame.columns:
        return None
    values = pd.to_numeric(frame[return_column], errors="coerce")
    valid = values.notna()
    return None if not valid.any() else float(values.loc[valid].gt(0).mean())


def _daily_metrics(
    window_id: str,
    as_of_date: str,
    group: pd.DataFrame,
    return_column: str,
    excess_column: str,
    top_pct: float,
    bottom_pct: float,
) -> dict[str, Any]:
    ordered = group.sort_values("predicted_probability", kind="stable").copy()
    count = len(ordered)
    top_count = max(1, int(math.floor(count * top_pct)))
    bottom_count = max(1, int(math.floor(count * bottom_pct)))
    if top_count + bottom_count >= count:
        raise ValueError("daily top/bottom groups leave no middle samples")
    bottom = ordered.iloc[:bottom_count]
    top = ordered.iloc[count - top_count :]
    middle = ordered.iloc[bottom_count : count - top_count]
    top_return = _frame_mean(top, return_column)
    middle_return = _frame_mean(middle, return_column)
    bottom_return = _frame_mean(bottom, return_column)
    top_excess = _frame_mean(top, excess_column)
    bottom_excess = _frame_mean(bottom, excess_column)
    return {
        "window_id": window_id,
        "as_of_date": as_of_date,
        "sample_count": count,
        "daily_spearman": _correlation(
            group["predicted_probability"], group[return_column]
        ),
        "daily_top_avg_return": top_return,
        "daily_middle_avg_return": middle_return,
        "daily_bottom_avg_return": bottom_return,
        "daily_top_bottom_spread": (
            None if top_return is None or bottom_return is None
            else top_return - bottom_return
        ),
        "daily_top_avg_excess_return": top_excess,
        "daily_bottom_avg_excess_return": bottom_excess,
        "daily_top_bottom_excess_spread": (
            None if top_excess is None or bottom_excess is None
            else top_excess - bottom_excess
        ),
        "daily_top_hit_rate": _hit_rate(top, return_column),
        "daily_bottom_hit_rate": _hit_rate(bottom, return_column),
    }


def _status(row: dict[str, Any]) -> str:
    spearman = _number(row.get("avg_daily_spearman"))
    spread = _number(row.get("avg_daily_top_bottom_spread"))
    top = _number(row.get("avg_daily_top_return"))
    middle = _number(row.get("avg_daily_middle_return"))
    bottom = _number(row.get("avg_daily_bottom_return"))
    if spearman is None or spread is None or spearman <= 0 or spread <= 0:
        return "failed"
    if (
        top is not None and middle is not None and bottom is not None
        and top > middle > bottom
    ):
        return "positive_structure"
    return "weak_positive"


def _empty_window(
    window_id: str,
    train_end: pd.Timestamp,
    valid_start: pd.Timestamp,
    valid_end: pd.Timestamp,
    valid_samples: int,
) -> dict[str, Any]:
    row = {column: None for column in WINDOW_COLUMNS}
    row.update({
        "window_id": window_id,
        "train_end": train_end.date().isoformat(),
        "valid_start": valid_start.date().isoformat(),
        "valid_end": valid_end.date().isoformat(),
        "valid_days": 0,
        "valid_samples": valid_samples,
        "positive_day_rate": 0.0,
        "status": "failed",
    })
    return row


def _aggregate_window(
    window_id: str,
    train_end: pd.Timestamp,
    valid_start: pd.Timestamp,
    valid_end: pd.Timestamp,
    valid_samples: int,
    daily_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    row = _empty_window(window_id, train_end, valid_start, valid_end, valid_samples)
    if not daily_rows:
        return row
    daily = pd.DataFrame(daily_rows)
    row.update({
        "valid_days": len(daily),
        "avg_daily_spearman": _mean(daily["daily_spearman"]),
        "median_daily_spearman": _median(daily["daily_spearman"]),
        "avg_daily_top_return": _mean(daily["daily_top_avg_return"]),
        "avg_daily_middle_return": _mean(daily["daily_middle_avg_return"]),
        "avg_daily_bottom_return": _mean(daily["daily_bottom_avg_return"]),
        "avg_daily_top_bottom_spread": _mean(daily["daily_top_bottom_spread"]),
        "median_daily_top_bottom_spread": _median(daily["daily_top_bottom_spread"]),
        "avg_daily_top_excess_return": _mean(daily["daily_top_avg_excess_return"]),
        "avg_daily_bottom_excess_return": _mean(daily["daily_bottom_avg_excess_return"]),
        "avg_daily_top_bottom_excess_spread": _mean(
            daily["daily_top_bottom_excess_spread"]
        ),
        "avg_daily_top_hit_rate": _mean(daily["daily_top_hit_rate"]),
        "avg_daily_bottom_hit_rate": _mean(daily["daily_bottom_hit_rate"]),
        "positive_day_rate": float(
            pd.to_numeric(daily["daily_top_bottom_spread"], errors="coerce").gt(0).mean()
        ),
    })
    row["status"] = _status(row)
    return row


def _evaluate_window(
    frame: pd.DataFrame,
    dates: pd.Series,
    target_values: pd.Series,
    target: str,
    return_column: str,
    excess_column: str,
    train_end: pd.Timestamp,
    valid_start: pd.Timestamp,
    valid_end: pd.Timestamp,
    top_pct: float,
    bottom_pct: float,
    min_train_samples: int,
    min_valid_samples: int,
    min_daily_samples: int,
    window_id: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    train = frame.loc[dates.le(train_end) & target_values.notna()].copy()
    valid = frame.loc[
        dates.ge(valid_start) & dates.le(valid_end) & target_values.notna()
    ].copy()
    empty = _empty_window(window_id, train_end, valid_start, valid_end, len(valid))
    if len(train) < min_train_samples or len(valid) < min_valid_samples:
        return empty, []
    preprocessor = MLPreprocessor()
    features = preprocessor.select_features(train, target)
    y_train = pd.to_numeric(train[target], errors="coerce").astype(int)
    if not features or y_train.nunique() < 2:
        return empty, []
    pipeline = preprocessor.build_pipeline("logistic")
    try:
        pipeline.fit(preprocessor.numeric_frame(train, features), y_train)
        valid["predicted_probability"] = pipeline.predict_proba(
            preprocessor.numeric_frame(valid, features)
        )[:, 1]
    except Exception:
        return empty, []

    daily_rows: list[dict[str, Any]] = []
    for as_of_date, group in valid.groupby("as_of_date", sort=True):
        if len(group) < min_daily_samples:
            continue
        daily_rows.append(_daily_metrics(
            window_id=window_id,
            as_of_date=str(as_of_date),
            group=group,
            return_column=return_column,
            excess_column=excess_column,
            top_pct=top_pct,
            bottom_pct=bottom_pct,
        ))
    return (
        _aggregate_window(
            window_id, train_end, valid_start, valid_end, len(valid), daily_rows
        ),
        daily_rows,
    )


def _summary(args: argparse.Namespace, windows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(row["status"] for row in windows)
    positive = int(counts.get("positive_structure", 0))
    weak = int(counts.get("weak_positive", 0))
    failed = int(counts.get("failed", 0))
    count = len(windows)
    return {
        "experiment_name": args.experiment_name,
        "dataset": args.dataset,
        "target": args.target,
        "model": args.model,
        "window_count": count,
        "positive_structure_count": positive,
        "weak_positive_count": weak,
        "failed_count": failed,
        "pass_rate": 0.0 if count == 0 else (positive + weak) / count,
        "avg_daily_spearman_all_windows": _mean([
            row.get("avg_daily_spearman") for row in windows
        ]),
        "avg_daily_top_bottom_spread_all_windows": _mean([
            row.get("avg_daily_top_bottom_spread") for row in windows
        ]),
        "median_daily_top_bottom_spread_all_windows": _median([
            row.get("avg_daily_top_bottom_spread") for row in windows
        ]),
        "avg_positive_day_rate": _mean([
            row.get("positive_day_rate") for row in windows
        ]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Daily cross-sectional ML validation")
    parser.add_argument(
        "--dataset",
        default="data/ml/ml_research_dataset_liquid500_context_relative_2025_2026.csv",
    )
    parser.add_argument("--target", default="future_top30_5d")
    parser.add_argument("--model", default="logistic", choices=["logistic"])
    parser.add_argument(
        "--experiment-name", default="daily_cs_liquid500_context_top30_5d"
    )
    parser.add_argument("--initial-train-end", type=_date, default=_date("2025-09-30"))
    parser.add_argument("--valid-window-months", type=int, default=1)
    parser.add_argument("--step-months", type=int, default=1)
    parser.add_argument("--final-valid-end", type=_date, default=_date("2026-06-20"))
    parser.add_argument("--top-pct", type=float, default=0.20)
    parser.add_argument("--bottom-pct", type=float, default=0.20)
    parser.add_argument("--min-train-samples", type=int, default=5000)
    parser.add_argument("--min-valid-samples", type=int, default=500)
    parser.add_argument("--min-daily-samples", type=int, default=30)
    parser.add_argument("--output-dir", default="data/ml/walk_forward/daily_cross_sectional")
    args = parser.parse_args()

    if args.valid_window_months <= 0 or args.step_months <= 0:
        parser.error("window and step months must be positive")
    if not 0 < args.top_pct < 1 or not 0 < args.bottom_pct < 1:
        parser.error("top-pct and bottom-pct must be between 0 and 1")
    if args.top_pct + args.bottom_pct >= 1:
        parser.error("top-pct plus bottom-pct must be less than 1")
    if min(args.min_train_samples, args.min_valid_samples, args.min_daily_samples) <= 0:
        parser.error("minimum sample counts must be positive")
    if args.initial_train_end >= args.final_valid_end:
        parser.error("initial-train-end must be earlier than final-valid-end")
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        parser.error(f"dataset not found: {dataset_path}")
    frame = pd.read_csv(dataset_path, encoding="utf-8-sig")
    if args.target not in frame.columns or "as_of_date" not in frame.columns:
        parser.error("dataset must contain target and as_of_date")
    horizon_match = re.search(r"_(\d+)d$", args.target)
    if not horizon_match:
        parser.error("target must end with a horizon such as _5d")
    horizon = int(horizon_match.group(1))
    return_column = f"future_return_{horizon}d"
    excess_column = f"future_excess_return_{horizon}d"
    if return_column not in frame.columns or excess_column not in frame.columns:
        parser.error("dataset is missing future return or excess return labels")

    dates = pd.to_datetime(frame["as_of_date"], errors="coerce")
    target_values = pd.to_numeric(frame[args.target], errors="coerce")
    train_end = pd.Timestamp(args.initial_train_end)
    final_valid_end = pd.Timestamp(args.final_valid_end)
    windows: list[dict[str, Any]] = []
    daily_rows: list[dict[str, Any]] = []
    number = 1
    while train_end < final_valid_end:
        valid_start = train_end + pd.Timedelta(days=1)
        valid_end = min(
            valid_start + pd.DateOffset(months=args.valid_window_months) - pd.Timedelta(days=1),
            final_valid_end,
        )
        window, daily = _evaluate_window(
            frame, dates, target_values, args.target, return_column, excess_column,
            train_end, valid_start, valid_end, args.top_pct, args.bottom_pct,
            args.min_train_samples, args.min_valid_samples, args.min_daily_samples,
            f"window_{number:02d}",
        )
        windows.append(window)
        daily_rows.extend(daily)
        next_train_end = valid_start + pd.DateOffset(months=args.step_months) - pd.Timedelta(days=1)
        if next_train_end <= train_end:
            break
        train_end = next_train_end
        number += 1

    summary = _summary(args, windows)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_daily = output_dir / f"{args.experiment_name}_daily.csv"
    output_windows = output_dir / f"{args.experiment_name}_windows.csv"
    output_json = output_dir / f"{args.experiment_name}_summary.json"
    pd.DataFrame(daily_rows, columns=DAILY_COLUMNS).to_csv(
        output_daily, index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(windows, columns=WINDOW_COLUMNS).to_csv(
        output_windows, index=False, encoding="utf-8-sig"
    )
    output_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )

    print("StockLens Daily Cross-sectional Validation")
    print(f"windows: {summary['window_count']}")
    print(f"positive_structure: {summary['positive_structure_count']}")
    print(f"weak_positive: {summary['weak_positive_count']}")
    print(f"failed: {summary['failed_count']}")
    print(f"pass_rate: {summary['pass_rate']}")
    print(f"avg_daily_spearman: {summary['avg_daily_spearman_all_windows']}")
    print(
        "avg_daily_top_bottom_spread: "
        f"{summary['avg_daily_top_bottom_spread_all_windows']}"
    )
    print(f"avg_positive_day_rate: {summary['avg_positive_day_rate']}")
    print(f"output_daily: {output_daily}")
    print(f"output_windows: {output_windows}")
    print(f"output_json: {output_json}")


if __name__ == "__main__":
    main()
