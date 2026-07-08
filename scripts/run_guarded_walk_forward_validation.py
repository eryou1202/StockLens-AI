from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.run_walk_forward_validation import _bucket_value, _mean, _structure_status
from src.ml.ml_evaluator import MLEvaluator
from src.ml.ml_preprocess import MLPreprocessor
from src.ml.ml_scoring_guard import MLScoringGuard


WINDOW_COLUMNS = [
    "window_id", "valid_start", "valid_end", "raw_spearman", "guarded_spearman",
    "raw_top_bottom_spread", "guarded_top_bottom_spread",
    "raw_top_bottom_excess_spread", "guarded_top_bottom_excess_spread",
    "raw_status", "guarded_status", "guard_improved", "guarded_top20_avg_return",
    "guarded_middle60_avg_return", "guarded_bottom20_avg_return",
    "guarded_top20_hit_rate", "guarded_bottom20_hit_rate", "top20_guard_flag_counts",
]


def _date(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d")


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if np.isfinite(result) else None


def _rank_top_indices(scores: np.ndarray, index: pd.Index) -> pd.Index:
    work = pd.DataFrame({"score": scores}, index=index).dropna(subset=["score"])
    work = work.sort_values("score", kind="stable")
    count = len(work)
    if count == 0:
        return pd.Index([])
    edge = max(1, int(math.floor(count * 0.20)))
    if edge * 2 >= count:
        edge = max(1, count // 3)
    return work.index[count - edge :]


def _guard_effect(raw_status: str, guarded_status: str, raw_spread: Any, guarded_spread: Any) -> str:
    levels = {"failed": 0, "weak_positive": 1, "positive_structure": 2}
    raw_level = levels.get(raw_status, 0)
    guarded_level = levels.get(guarded_status, 0)
    if guarded_level > raw_level:
        return "improved"
    if guarded_level < raw_level:
        return "worsened"
    raw_value = _number(raw_spread)
    guarded_value = _number(guarded_spread)
    if raw_value is None or guarded_value is None or math.isclose(raw_value, guarded_value, abs_tol=1e-12):
        return "unchanged"
    return "improved" if guarded_value > raw_value else "worsened"


def _failed_row(
    window_id: str,
    valid_start: pd.Timestamp,
    valid_end: pd.Timestamp,
) -> dict[str, Any]:
    row = {column: None for column in WINDOW_COLUMNS}
    row.update({
        "window_id": window_id,
        "valid_start": valid_start.date().isoformat(),
        "valid_end": valid_end.date().isoformat(),
        "raw_status": "failed",
        "guarded_status": "failed",
        "guard_improved": False,
        "top20_guard_flag_counts": "{}",
        "_guard_effect": "unchanged",
    })
    return row


def _evaluate_window(
    frame: pd.DataFrame,
    dates: pd.Series,
    target_values: pd.Series,
    target: str,
    train_end: pd.Timestamp,
    valid_start: pd.Timestamp,
    valid_end: pd.Timestamp,
    min_train_samples: int,
    min_valid_samples: int,
    window_id: str,
) -> dict[str, Any]:
    train = frame.loc[dates.le(train_end) & target_values.notna()].copy()
    valid = frame.loc[
        dates.ge(valid_start) & dates.le(valid_end) & target_values.notna()
    ].copy()
    row = _failed_row(window_id, valid_start, valid_end)
    if len(train) < min_train_samples or len(valid) < min_valid_samples:
        return row

    preprocessor = MLPreprocessor()
    features = preprocessor.select_features(train, target)
    y_train = pd.to_numeric(train[target], errors="coerce").astype(int)
    if not features or y_train.nunique() < 2:
        return row
    pipeline = preprocessor.build_pipeline("logistic")
    try:
        pipeline.fit(preprocessor.numeric_frame(train, features), y_train)
        raw_scores = pipeline.predict_proba(
            preprocessor.numeric_frame(valid, features)
        )[:, 1]
    except Exception:
        return row

    guard = MLScoringGuard()
    guard_results = [guard.apply(sample, raw) for (_, sample), raw in zip(valid.iterrows(), raw_scores)]
    guarded_scores = np.array([item.guarded_score for item in guard_results], dtype=float)
    guard_actions = [item.guard_action for item in guard_results]
    guarded_rank_scores = np.array([
        -1.0 if item.guard_action == "block" else item.guarded_score
        for item in guard_results
    ], dtype=float)

    evaluator = MLEvaluator()
    raw_metrics = evaluator.evaluate_classification(
        valid, target, raw_scores, (raw_scores >= 0.5).astype(int)
    )
    guarded_metrics = evaluator.evaluate_classification(
        valid,
        target,
        guarded_rank_scores,
        np.array([
            int(score >= 0.5 and action != "block")
            for score, action in zip(guarded_scores, guard_actions)
        ]),
    )
    raw_status = _structure_status(raw_metrics)
    guarded_status = _structure_status(guarded_metrics)
    effect = _guard_effect(
        raw_status,
        guarded_status,
        raw_metrics.get("top_bottom_spread"),
        guarded_metrics.get("top_bottom_spread"),
    )
    top_indices = _rank_top_indices(guarded_rank_scores, valid.index)
    index_to_result = {
        index: result for index, result in zip(valid.index, guard_results)
    }
    flag_counts = Counter(
        flag
        for index in top_indices
        for flag in index_to_result[index].guard_flags
    )
    row.update({
        "raw_spearman": raw_metrics.get("spearman_corr_with_future_return"),
        "guarded_spearman": guarded_metrics.get("spearman_corr_with_future_return"),
        "raw_top_bottom_spread": raw_metrics.get("top_bottom_spread"),
        "guarded_top_bottom_spread": guarded_metrics.get("top_bottom_spread"),
        "raw_top_bottom_excess_spread": raw_metrics.get("top_bottom_excess_spread"),
        "guarded_top_bottom_excess_spread": guarded_metrics.get("top_bottom_excess_spread"),
        "raw_status": raw_status,
        "guarded_status": guarded_status,
        "guard_improved": effect == "improved",
        "guarded_top20_avg_return": guarded_metrics.get("top20_avg_return"),
        "guarded_middle60_avg_return": guarded_metrics.get("middle60_avg_return"),
        "guarded_bottom20_avg_return": guarded_metrics.get("bottom20_avg_return"),
        "guarded_top20_hit_rate": _bucket_value(guarded_metrics, "top20", "hit_rate"),
        "guarded_bottom20_hit_rate": _bucket_value(guarded_metrics, "bottom20", "hit_rate"),
        "top20_guard_flag_counts": json.dumps(
            dict(sorted(flag_counts.items())), ensure_ascii=False
        ),
        "_guard_effect": effect,
    })
    return row


def _build_summary(args: argparse.Namespace, rows: list[dict[str, Any]]) -> dict[str, Any]:
    raw_pass = sum(row["raw_status"] in {"positive_structure", "weak_positive"} for row in rows)
    guarded_pass = sum(
        row["guarded_status"] in {"positive_structure", "weak_positive"} for row in rows
    )
    effects = Counter(row.get("_guard_effect", "unchanged") for row in rows)
    count = len(rows)
    return {
        "experiment_name": args.experiment_name,
        "dataset": args.dataset,
        "target": args.target,
        "model": args.model,
        "window_count": count,
        "raw_pass_rate": 0.0 if count == 0 else raw_pass / count,
        "guarded_pass_rate": 0.0 if count == 0 else guarded_pass / count,
        "raw_avg_spearman": _mean([row.get("raw_spearman") for row in rows]),
        "guarded_avg_spearman": _mean([row.get("guarded_spearman") for row in rows]),
        "raw_avg_spread": _mean([row.get("raw_top_bottom_spread") for row in rows]),
        "guarded_avg_spread": _mean([
            row.get("guarded_top_bottom_spread") for row in rows
        ]),
        "improved_windows": int(effects.get("improved", 0)),
        "worsened_windows": int(effects.get("worsened", 0)),
        "unchanged_windows": int(effects.get("unchanged", 0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Guarded StockLens walk-forward validation")
    parser.add_argument(
        "--dataset",
        default="data/ml/ml_research_dataset_liquid500_context_relative_2025_2026.csv",
    )
    parser.add_argument("--target", default="future_top30_5d")
    parser.add_argument("--model", default="logistic", choices=["logistic"])
    parser.add_argument(
        "--experiment-name", default="guarded_walk_forward_liquid500_context_top30_5d"
    )
    parser.add_argument("--initial-train-end", type=_date, default=_date("2025-09-30"))
    parser.add_argument("--valid-window-months", type=int, default=1)
    parser.add_argument("--step-months", type=int, default=1)
    parser.add_argument("--final-valid-end", type=_date, default=_date("2026-06-20"))
    parser.add_argument("--min-train-samples", type=int, default=5000)
    parser.add_argument("--min-valid-samples", type=int, default=500)
    parser.add_argument("--output-dir", default="data/ml/walk_forward/guarded")
    args = parser.parse_args()

    if args.valid_window_months <= 0 or args.step_months <= 0:
        parser.error("window and step months must be positive")
    if args.min_train_samples <= 0 or args.min_valid_samples <= 0:
        parser.error("minimum sample counts must be positive")
    if args.initial_train_end >= args.final_valid_end:
        parser.error("initial-train-end must be earlier than final-valid-end")
    path = Path(args.dataset)
    if not path.exists():
        parser.error(f"dataset not found: {path}")
    frame = pd.read_csv(path, encoding="utf-8-sig")
    if args.target not in frame.columns or "as_of_date" not in frame.columns:
        parser.error("dataset must contain target and as_of_date")

    dates = pd.to_datetime(frame["as_of_date"], errors="coerce")
    target_values = pd.to_numeric(frame[args.target], errors="coerce")
    train_end = pd.Timestamp(args.initial_train_end)
    final_valid_end = pd.Timestamp(args.final_valid_end)
    rows: list[dict[str, Any]] = []
    number = 1
    while train_end < final_valid_end:
        valid_start = train_end + pd.Timedelta(days=1)
        valid_end = min(
            valid_start + pd.DateOffset(months=args.valid_window_months) - pd.Timedelta(days=1),
            final_valid_end,
        )
        rows.append(_evaluate_window(
            frame, dates, target_values, args.target, train_end, valid_start, valid_end,
            args.min_train_samples, args.min_valid_samples, f"window_{number:02d}",
        ))
        next_train_end = valid_start + pd.DateOffset(months=args.step_months) - pd.Timedelta(days=1)
        if next_train_end <= train_end:
            break
        train_end = next_train_end
        number += 1

    summary = _build_summary(args, rows)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / f"{args.experiment_name}_windows.csv"
    output_json = output_dir / f"{args.experiment_name}_summary.json"
    pd.DataFrame(rows, columns=WINDOW_COLUMNS).to_csv(
        output_csv, index=False, encoding="utf-8-sig"
    )
    output_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )

    print("StockLens Guarded Walk-forward Validation")
    print(f"windows: {summary['window_count']}")
    print(f"raw_pass_rate: {summary['raw_pass_rate']}")
    print(f"guarded_pass_rate: {summary['guarded_pass_rate']}")
    print(f"raw_avg_spearman: {summary['raw_avg_spearman']}")
    print(f"guarded_avg_spearman: {summary['guarded_avg_spearman']}")
    print(f"raw_avg_top_bottom_spread: {summary['raw_avg_spread']}")
    print(f"guarded_avg_top_bottom_spread: {summary['guarded_avg_spread']}")
    print(f"improved_windows: {summary['improved_windows']}")
    print(f"worsened_windows: {summary['worsened_windows']}")
    print(f"output_csv: {output_csv}")
    print(f"output_json: {output_json}")


if __name__ == "__main__":
    main()
