from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.ml.ml_evaluator import MLEvaluator
from src.ml.ml_preprocess import MLPreprocessor
from src.ml.ml_trainer import MLTrainer


WINDOW_COLUMNS = [
    "window_id", "train_start", "train_end", "valid_start", "valid_end",
    "train_samples", "valid_samples", "accuracy", "precision", "recall", "roc_auc",
    "pearson_corr_with_future_return", "spearman_corr_with_future_return",
    "top20_avg_return", "middle60_avg_return", "bottom20_avg_return",
    "top_bottom_spread", "top20_avg_excess_return", "middle60_avg_excess_return",
    "bottom20_avg_excess_return", "top_bottom_excess_spread",
    "top20_hit_rate", "bottom20_hit_rate", "status",
]


def _date(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d")


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if np.isfinite(result) else None


def _mean(values: list[Any]) -> float | None:
    numeric = [_number(value) for value in values]
    valid = [value for value in numeric if value is not None]
    return None if not valid else float(np.mean(valid))


def _median(values: list[Any]) -> float | None:
    numeric = [_number(value) for value in values]
    valid = [value for value in numeric if value is not None]
    return None if not valid else float(np.median(valid))


def _bucket_value(metrics: dict[str, Any], bucket: str, field: str) -> float | None:
    for item in metrics.get("buckets", []):
        if item.get("bucket") == bucket:
            return _number(item.get(field))
    return None


def _structure_status(metrics: dict[str, Any]) -> str:
    spearman = _number(metrics.get("spearman_corr_with_future_return"))
    spread = _number(metrics.get("top_bottom_spread"))
    excess_spread = _number(metrics.get("top_bottom_excess_spread"))
    top = _number(metrics.get("top20_avg_return"))
    middle = _number(metrics.get("middle60_avg_return"))
    bottom = _number(metrics.get("bottom20_avg_return"))
    if spearman is None or spread is None or spearman <= 0 or spread <= 0:
        return "failed"
    monotonic = (
        top is not None and middle is not None and bottom is not None
        and top > middle > bottom
    )
    if excess_spread is not None and excess_spread > 0 and monotonic:
        return "positive_structure"
    return "weak_positive"


def _empty_window(
    window_id: str,
    train_start: str | None,
    train_end: pd.Timestamp,
    valid_start: pd.Timestamp,
    valid_end: pd.Timestamp,
    train_samples: int,
    valid_samples: int,
) -> dict[str, Any]:
    row = {column: None for column in WINDOW_COLUMNS}
    row.update({
        "window_id": window_id,
        "train_start": train_start,
        "train_end": train_end.date().isoformat(),
        "valid_start": valid_start.date().isoformat(),
        "valid_end": valid_end.date().isoformat(),
        "train_samples": train_samples,
        "valid_samples": valid_samples,
        "status": "failed",
    })
    return row


def _evaluate_window(
    frame: pd.DataFrame,
    dates: pd.Series,
    target_values: pd.Series,
    target: str,
    model: str,
    train_end: pd.Timestamp,
    valid_start: pd.Timestamp,
    valid_end: pd.Timestamp,
    min_train_samples: int,
    min_valid_samples: int,
    window_id: str,
) -> dict[str, Any]:
    train_mask = dates.le(train_end) & target_values.notna()
    valid_mask = dates.ge(valid_start) & dates.le(valid_end) & target_values.notna()
    train = frame.loc[train_mask].copy()
    valid = frame.loc[valid_mask].copy()
    train_start = (
        pd.to_datetime(train["as_of_date"], errors="coerce").min().date().isoformat()
        if not train.empty else None
    )
    row = _empty_window(
        window_id, train_start, train_end, valid_start, valid_end, len(train), len(valid)
    )
    if len(train) < min_train_samples or len(valid) < min_valid_samples:
        return row

    preprocessor = MLPreprocessor()
    evaluator = MLEvaluator()
    features = preprocessor.select_features(train, target)
    if not features:
        return row
    task = MLTrainer._task_type(target)
    y_train = pd.to_numeric(train[target], errors="coerce")
    if task == "classification":
        y_train = y_train.astype(int)
        if y_train.nunique() < 2:
            return row
    x_train = preprocessor.numeric_frame(train, features)
    x_valid = preprocessor.numeric_frame(valid, features)
    pipeline = preprocessor.build_pipeline(model)
    try:
        pipeline.fit(x_train, y_train)
        predictions = pipeline.predict(x_valid)
        if task == "classification":
            probabilities = pipeline.predict_proba(x_valid)[:, 1]
            metrics = evaluator.evaluate_classification(
                valid, target, probabilities, predictions
            )
        else:
            metrics = evaluator.evaluate_regression(valid, target, predictions)
    except Exception:
        return row

    for key in (
        "accuracy", "precision", "recall", "roc_auc",
        "pearson_corr_with_future_return", "spearman_corr_with_future_return",
        "top20_avg_return", "middle60_avg_return", "bottom20_avg_return",
        "top_bottom_spread", "top20_avg_excess_return",
        "middle60_avg_excess_return", "bottom20_avg_excess_return",
        "top_bottom_excess_spread",
    ):
        row[key] = metrics.get(key)
    row["top20_hit_rate"] = _bucket_value(metrics, "top20", "hit_rate")
    row["bottom20_hit_rate"] = _bucket_value(metrics, "bottom20", "hit_rate")
    row["status"] = _structure_status(metrics)
    return row


def _build_summary(
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    counts = pd.Series([row["status"] for row in rows], dtype="object").value_counts()
    positive = int(counts.get("positive_structure", 0))
    weak = int(counts.get("weak_positive", 0))
    failed = int(counts.get("failed", 0))
    window_count = len(rows)
    spearman = [row.get("spearman_corr_with_future_return") for row in rows]
    spread = [row.get("top_bottom_spread") for row in rows]
    excess_spread = [row.get("top_bottom_excess_spread") for row in rows]
    return {
        "experiment_name": args.experiment_name,
        "dataset": args.dataset,
        "target": args.target,
        "model": args.model,
        "window_count": window_count,
        "positive_structure_count": positive,
        "weak_positive_count": weak,
        "failed_count": failed,
        "avg_spearman": _mean(spearman),
        "median_spearman": _median(spearman),
        "avg_top_bottom_spread": _mean(spread),
        "median_top_bottom_spread": _median(spread),
        "avg_top_bottom_excess_spread": _mean(excess_spread),
        "median_top_bottom_excess_spread": _median(excess_spread),
        "pass_rate": 0.0 if window_count == 0 else (positive + weak) / window_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="StockLens expanding walk-forward validation")
    parser.add_argument(
        "--dataset",
        default="data/ml/ml_research_dataset_liquid500_context_relative_2025_2026.csv",
    )
    parser.add_argument("--target", default="future_top30_5d")
    parser.add_argument(
        "--model", default="logistic", choices=["logistic", "random_forest_regressor"]
    )
    parser.add_argument(
        "--experiment-name", default="walk_forward_liquid500_context_top30_5d"
    )
    parser.add_argument("--initial-train-end", type=_date, default=_date("2025-09-30"))
    parser.add_argument("--valid-window-months", type=int, default=1)
    parser.add_argument("--step-months", type=int, default=1)
    parser.add_argument("--final-valid-end", type=_date, default=_date("2026-06-20"))
    parser.add_argument("--min-train-samples", type=int, default=5000)
    parser.add_argument("--min-valid-samples", type=int, default=500)
    parser.add_argument("--output-dir", default="data/ml/walk_forward")
    args = parser.parse_args()

    if args.valid_window_months <= 0 or args.step_months <= 0:
        parser.error("window and step months must be positive")
    if args.min_train_samples <= 0 or args.min_valid_samples <= 0:
        parser.error("minimum sample counts must be positive")
    if args.initial_train_end >= args.final_valid_end:
        parser.error("initial-train-end must be earlier than final-valid-end")
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        parser.error(f"dataset not found: {dataset_path}")

    frame = pd.read_csv(dataset_path, encoding="utf-8-sig")
    if args.target not in frame.columns or "as_of_date" not in frame.columns:
        parser.error("dataset must contain target and as_of_date")
    task = MLTrainer._task_type(args.target)
    if task == "classification" and args.model != "logistic":
        parser.error("classification target requires logistic")
    if task == "regression" and args.model != "random_forest_regressor":
        parser.error("regression target requires random_forest_regressor")

    dates = pd.to_datetime(frame["as_of_date"], errors="coerce")
    target_values = pd.to_numeric(frame[args.target], errors="coerce")
    train_end = pd.Timestamp(args.initial_train_end)
    final_valid_end = pd.Timestamp(args.final_valid_end)
    rows: list[dict[str, Any]] = []
    window_number = 1
    while train_end < final_valid_end:
        valid_start = train_end + pd.Timedelta(days=1)
        valid_end = min(
            valid_start + pd.DateOffset(months=args.valid_window_months) - pd.Timedelta(days=1),
            final_valid_end,
        )
        rows.append(_evaluate_window(
            frame=frame,
            dates=dates,
            target_values=target_values,
            target=args.target,
            model=args.model,
            train_end=train_end,
            valid_start=valid_start,
            valid_end=valid_end,
            min_train_samples=args.min_train_samples,
            min_valid_samples=args.min_valid_samples,
            window_id=f"window_{window_number:02d}",
        ))
        next_train_end = (
            valid_start + pd.DateOffset(months=args.step_months) - pd.Timedelta(days=1)
        )
        if next_train_end <= train_end:
            break
        train_end = next_train_end
        window_number += 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / f"{args.experiment_name}_windows.csv"
    output_json = output_dir / f"{args.experiment_name}_summary.json"
    pd.DataFrame(rows, columns=WINDOW_COLUMNS).to_csv(
        output_csv, index=False, encoding="utf-8-sig"
    )
    summary = _build_summary(args, rows)
    output_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )

    print("StockLens Walk-forward Validation")
    print(f"experiment: {args.experiment_name}")
    print(f"dataset: {args.dataset}")
    print(f"target: {args.target}")
    print(f"model: {args.model}")
    print(f"windows: {summary['window_count']}")
    print(f"positive_structure: {summary['positive_structure_count']}")
    print(f"weak_positive: {summary['weak_positive_count']}")
    print(f"failed: {summary['failed_count']}")
    print(f"pass_rate: {summary['pass_rate']}")
    print(f"avg_spearman: {summary['avg_spearman']}")
    print(f"avg_top_bottom_spread: {summary['avg_top_bottom_spread']}")
    print(f"avg_top_bottom_excess_spread: {summary['avg_top_bottom_excess_spread']}")
    print(f"output_csv: {output_csv}")
    print(f"output_json: {output_json}")


if __name__ == "__main__":
    main()
