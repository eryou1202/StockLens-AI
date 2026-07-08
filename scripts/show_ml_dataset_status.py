from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


LABEL_PREFIXES = (
    "future_return_", "future_excess_return_", "future_rank_pct_",
    "future_top30_", "future_bottom30_", "hit_", "future_max_drawdown_",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="查看 ML 研究数据集状态")
    parser.add_argument("--dataset", default="data/ml/ml_research_dataset.csv")
    args = parser.parse_args()
    path = Path(args.dataset)
    if not path.exists():
        print(f"数据集不存在: {path}")
        return
    frame = pd.read_csv(path, encoding="utf-8-sig")
    feature_columns = [
        column for column in frame.columns
        if column not in {
            "sample_id", "symbol", "stock_name", "as_of_date", "price_time", "current_price",
            "source", "sample_interval_days", "lookback_days", "label_status", "label_error",
        }
        and not column.startswith(LABEL_PREFIXES)
    ]
    label_columns = [
        column for column in frame.columns
        if column.startswith(LABEL_PREFIXES)
    ]
    relative_label_columns = [
        column for column in frame.columns
        if column.startswith((
            "future_excess_return_", "future_rank_pct_",
            "future_top30_", "future_bottom30_",
        ))
    ]
    dates = pd.to_datetime(frame.get("as_of_date"), errors="coerce")
    print("StockLens ML Research Dataset Status")
    print(f"dataset: {path}")
    print(f"rows: {len(frame)}")
    print(f"symbols: {frame['symbol'].nunique() if 'symbol' in frame else 0}")
    print(f"date range: {dates.min().date() if dates.notna().any() else '-'} -> "
          f"{dates.max().date() if dates.notna().any() else '-'}")
    print(f"feature columns: {len(feature_columns)}")
    print(f"label columns: {len(label_columns)}")
    print(f"relative label columns: {len(relative_label_columns)}")
    if "label_status" in frame:
        print("label status:")
        for status, count in frame["label_status"].fillna("missing").value_counts().items():
            print(f"  {status}: {count}")
    print("label availability:")
    for column in label_columns:
        print(f"  {column}: {int(pd.to_numeric(frame[column], errors='coerce').notna().sum())}")
    print("relative 5d availability:")
    for column in (
        "future_excess_return_5d", "future_rank_pct_5d",
        "future_top30_5d", "future_bottom30_5d",
    ):
        available = (
            int(pd.to_numeric(frame[column], errors="coerce").notna().sum())
            if column in frame.columns else 0
        )
        print(f"  {column}: {available}")
    print("data boundary: feature builder=past-only; label builder=future-only")


if __name__ == "__main__":
    main()
