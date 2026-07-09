from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


OUTCOME_COLUMNS = [
    "future_return_1d", "future_return_3d", "future_return_5d",
    "future_return_10d", "future_excess_return_5d", "future_rank_pct_5d",
    "hit_5d", "outcome_status",
]


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if np.isfinite(result) else None


def _mean(series: pd.Series) -> float | None:
    values = pd.to_numeric(series, errors="coerce")
    value = values.mean()
    return None if pd.isna(value) else float(value)


def _group_metrics(frame: pd.DataFrame, dimension: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    grouped = frame.groupby(dimension, dropna=False, sort=True)
    for value, group in grouped:
        completed = pd.to_numeric(group["future_return_5d"], errors="coerce").notna()
        rows.append({
            dimension: None if pd.isna(value) else str(value),
            "total_signals": int(len(group)),
            "completed_5d_count": int(completed.sum()),
            "pending_count": int((~completed).sum()),
            "avg_return_1d": _mean(group["future_return_1d"]),
            "avg_return_3d": _mean(group["future_return_3d"]),
            "avg_return_5d": _mean(group["future_return_5d"]),
            "avg_return_10d": _mean(group["future_return_10d"]),
            "avg_excess_return_5d": _mean(group["future_excess_return_5d"]),
            "avg_rank_pct_5d": _mean(group["future_rank_pct_5d"]),
            "hit_rate_5d": _mean(group["hit_5d"]),
        })
    return rows


def _bucket_frame(frame: pd.DataFrame, bucket: str) -> pd.DataFrame:
    if bucket == "top10":
        values = {"top10"}
    elif bucket == "top20":
        values = {"top10", "top20"}
    elif bucket == "top50":
        values = {"top10", "top20", "top50"}
    else:
        values = {bucket}
    return frame.loc[frame["ml_bucket"].isin(values)]


def _summary(frame: pd.DataFrame, history_file: str, dataset: str) -> dict[str, Any]:
    completed = pd.to_numeric(frame["future_return_5d"], errors="coerce").notna()
    summary: dict[str, Any] = {
        "history_file": history_file,
        "dataset": dataset,
        "total_signals": int(len(frame)),
        "completed_5d_count": int(completed.sum()),
        "pending_count": int((~completed).sum()),
        "bucket_scope": {
            "top10": "ml_bucket == top10",
            "top20": "top10 + top20 cumulative",
            "top50": "top10 + top20 + top50 cumulative",
        },
    }
    for bucket in ("top10", "top20", "top50"):
        group = _bucket_frame(frame, bucket)
        summary[f"{bucket}_avg_return_5d"] = _mean(group["future_return_5d"])
        summary[f"{bucket}_hit_rate_5d"] = _mean(group["hit_5d"])
    for level in ("extreme", "high", "medium", "low"):
        group = frame.loc[frame["shadow_risk_level"].eq(level)]
        summary[f"{level}_avg_return_5d"] = _mean(group["future_return_5d"])
    summary["aggregations"] = {
        "by_ml_bucket": _group_metrics(frame, "ml_bucket"),
        "by_shadow_risk_level": _group_metrics(frame, "shadow_risk_level"),
        "by_shadow_action": _group_metrics(frame, "shadow_action"),
        "by_as_of_date": _group_metrics(frame, "as_of_date"),
    }
    summary["research_only_note"] = (
        "ML shadow outcomes are research statistics and do not affect formal recommendations."
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Update research-only ML shadow outcomes")
    parser.add_argument(
        "--history-file", default="data/ml/shadow_mode/ml_shadow_history.csv"
    )
    parser.add_argument(
        "--dataset",
        default="data/ml/ml_research_dataset_liquid500_context_relative_daily_latest.csv",
    )
    parser.add_argument("--output-dir", default="data/ml/shadow_mode")
    args = parser.parse_args()

    history_path = Path(args.history_file)
    dataset_path = Path(args.dataset)
    if not history_path.exists():
        parser.error(f"history file not found: {history_path}")
    if not dataset_path.exists():
        parser.error(f"dataset not found: {dataset_path}")
    history = pd.read_csv(history_path, encoding="utf-8-sig")
    dataset = pd.read_csv(dataset_path, encoding="utf-8-sig")
    required_history = {
        "as_of_date", "symbol", "ml_bucket", "shadow_risk_level", "shadow_action"
    }
    missing_history = sorted(required_history.difference(history.columns))
    if missing_history:
        parser.error(f"history missing columns: {missing_history}")
    required_dataset = {"as_of_date", "symbol"}
    missing_dataset = sorted(required_dataset.difference(dataset.columns))
    if missing_dataset:
        parser.error(f"dataset missing columns: {missing_dataset}")

    # Normalize merge keys without changing the values retained in the output.
    history["as_of_date"] = pd.to_datetime(
        history["as_of_date"], errors="coerce"
    ).dt.strftime("%Y-%m-%d")
    dataset["as_of_date"] = pd.to_datetime(
        dataset["as_of_date"], errors="coerce"
    ).dt.strftime("%Y-%m-%d")
    history["symbol"] = history["symbol"].astype("string").str.strip().str.upper()
    dataset["symbol"] = dataset["symbol"].astype("string").str.strip().str.upper()

    history = history.drop(columns=OUTCOME_COLUMNS, errors="ignore")
    label_columns = [column for column in OUTCOME_COLUMNS[:-1] if column in dataset.columns]
    labels = dataset[["as_of_date", "symbol", *label_columns]].copy()
    labels = labels.drop_duplicates(["as_of_date", "symbol"], keep="last")
    outcomes = history.merge(
        labels,
        on=["as_of_date", "symbol"],
        how="left",
        validate="many_to_one",
    )
    for column in OUTCOME_COLUMNS[:-1]:
        if column not in outcomes.columns:
            outcomes[column] = np.nan
        outcomes[column] = pd.to_numeric(outcomes[column], errors="coerce")
    missing_hit = outcomes["hit_5d"].isna() & outcomes["future_return_5d"].notna()
    outcomes.loc[missing_hit, "hit_5d"] = (
        outcomes.loc[missing_hit, "future_return_5d"].gt(0).astype(int)
    )
    outcomes["outcome_status"] = np.where(
        outcomes["future_return_5d"].notna(), "complete", "pending"
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / "ml_shadow_outcomes.csv"
    output_json = output_dir / "ml_shadow_outcome_summary.json"
    outcomes.to_csv(output_csv, index=False, encoding="utf-8-sig")
    summary = _summary(outcomes, args.history_file, args.dataset)
    output_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    print("StockLens ML Shadow Outcome Tracker")
    print(f"total_signals: {summary['total_signals']}")
    print(f"completed_5d_count: {summary['completed_5d_count']}")
    print(f"pending_count: {summary['pending_count']}")
    print(f"top10_avg_return_5d: {summary['top10_avg_return_5d']}")
    print(f"top20_avg_return_5d: {summary['top20_avg_return_5d']}")
    print(f"top50_avg_return_5d: {summary['top50_avg_return_5d']}")
    print(f"output_csv: {output_csv}")
    print(f"output_json: {output_json}")


if __name__ == "__main__":
    main()
