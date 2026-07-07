from __future__ import annotations

import argparse
from datetime import datetime

from src.audit.universe_loader import load_symbols_from_file, normalize_symbols
from src.config.settings import load_settings
from src.ml.ml_dataset_builder import MLDatasetBuilder
from src.ml.ml_dataset_store import MLDatasetStore
from src.ml.ml_schema import MLDatasetRequest


def _date(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d")


def main() -> None:
    parser = argparse.ArgumentParser(description="构建隔离的多周期 ML 研究数据集")
    parser.add_argument("--start", required=True, type=_date)
    parser.add_argument("--end", required=True, type=_date)
    parser.add_argument("--symbols", nargs="*", default=[])
    parser.add_argument("--symbols-file")
    parser.add_argument("--max-symbols", type=int)
    parser.add_argument("--lookback-days", type=int, default=120)
    parser.add_argument("--sample-interval-days", type=int, default=5)
    parser.add_argument("--horizons", nargs="+", type=int, default=[1, 2, 3, 5, 10, 20])
    parser.add_argument(
        "--include-context-features",
        action="store_true",
        help="加入仅使用 as_of_date 当日及以前数据的市场/宽度/相对强弱/风格特征",
    )
    parser.add_argument("--output", default="data/ml/ml_research_dataset.csv")
    args = parser.parse_args()

    symbols = list(args.symbols)
    if args.symbols_file:
        symbols.extend(load_symbols_from_file(args.symbols_file))
    symbols = normalize_symbols(symbols)
    if not symbols:
        parser.error("请通过 --symbols 或 --symbols-file 提供至少一只股票")
    if args.start > args.end:
        parser.error("--start 不能晚于 --end")

    request = MLDatasetRequest(
        start_date=args.start,
        end_date=args.end,
        symbols=symbols,
        max_symbols=args.max_symbols,
        lookback_days=args.lookback_days,
        sample_interval_days=args.sample_interval_days,
        horizons=args.horizons,
        include_context_features=args.include_context_features,
        output_path=args.output,
    )
    print("StockLens Multi-Horizon ML Research Dataset")
    print("特征严格只使用 as_of_date 当日及之前行情；未来行情仅用于标签。")
    print(f"context features: {'enabled' if args.include_context_features else 'disabled'}")
    samples = MLDatasetBuilder(load_settings()).build(request)
    frame = MLDatasetStore().save(samples, args.output)
    status = frame["label_status"].value_counts(dropna=False).to_dict()
    print(f"rows: {len(frame)}")
    print(f"symbols: {frame['symbol'].nunique() if not frame.empty else 0}")
    print(f"label_status: {status}")
    print(f"output: {args.output}")


if __name__ == "__main__":
    main()
