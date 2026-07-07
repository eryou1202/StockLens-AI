from __future__ import annotations

import argparse

from src.reference.symbol_reference import AShareSymbolReference


def main() -> None:
    parser = argparse.ArgumentParser(description="构建本地 A 股代码名称参考表")
    parser.add_argument("--symbols", nargs="*", default=[])
    parser.add_argument("--output", default="data/reference/a_share_symbols.csv")
    parser.add_argument("--no-provider", action="store_true", help="仅使用已有本地数据")
    parser.add_argument("--stock-only", action="store_true", help="输出仅保留股票")
    parser.add_argument(
        "--include-index",
        action="store_true",
        help="输出仅保留股票和指数；默认输出全部资产类型",
    )
    args = parser.parse_args()

    reference = AShareSymbolReference(reference_path=args.output)
    frame = reference.build(
        args.symbols,
        include_provider=not args.no_provider,
        stock_only=args.stock_only,
        include_index=args.include_index,
    )
    named = int(frame["stock_name"].notna().sum()) if "stock_name" in frame else 0
    print("StockLens A-share Symbol Reference")
    print(f"rows: {len(frame)}")
    print(f"named rows: {named}")
    if "asset_type" in frame:
        print(f"asset types: {frame['asset_type'].value_counts(dropna=False).to_dict()}")
    print(f"output: {reference.reference_path}")


if __name__ == "__main__":
    main()
