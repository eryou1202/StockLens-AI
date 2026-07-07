from __future__ import annotations

import argparse

import pandas as pd

from src.config.settings import load_settings
from src.scan.a_share_coarse_scanner import AShareCoarseScanner


def main() -> None:
    parser = argparse.ArgumentParser(description="StockLens A 股规则量化粗扫（不使用 ML）")
    parser.add_argument("--symbols-file")
    parser.add_argument("--max-symbols", type=int)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--min-avg-amount-20d", type=float, default=30_000_000.0)
    parser.add_argument("--include-risky", action="store_true")
    parser.add_argument("--output-dir", default="data/scans")
    parser.add_argument("--save-json", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-csv", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.max_symbols is not None and args.max_symbols <= 0:
        parser.error("--max-symbols 必须大于 0")
    if args.limit <= 0:
        parser.error("--limit 必须大于 0")
    if args.min_avg_amount_20d < 0:
        parser.error("--min-avg-amount-20d 不能为负数")

    scanner = AShareCoarseScanner(load_settings())
    print("StockLens A 股量化粗扫")
    print("scan type: rule-based coarse scan")
    print("ML used: False")
    result = scanner.run(
        symbols_file=args.symbols_file,
        max_symbols=args.max_symbols,
        limit=args.limit,
        min_avg_amount_20d=args.min_avg_amount_20d,
        include_risky=args.include_risky,
    )
    paths = scanner.save_results(
        result,
        output_dir=args.output_dir,
        save_json=args.save_json,
        save_csv=args.save_csv,
    )
    print(f"universe_count: {result['universe_count']}")
    print(f"scanned_count: {result['scanned_count']}")
    print(f"excluded_count: {result['excluded_count']}")
    print(f"latest_as_of_date: {result['latest_as_of_date']}")
    print(f"excluded_summary: {result['excluded_summary']}")
    top = pd.DataFrame(result["top_candidates"])
    if top.empty:
        print("Top candidates: none")
    else:
        columns = [
            "rank", "symbol", "stock_name", "current_price", "coarse_score",
            "quant_score", "quant_decision", "return_5d", "return_20d",
            "risk_score", "overheat_score", "amount_ratio_5d",
        ]
        print(f"Top {len(top)} candidates:")
        print(top[columns].to_string(index=False))
    if args.include_risky:
        print(f"risk_candidates: {len(result['risk_candidates'])}")
    for name, path in paths.items():
        print(f"{name}: {path}")
    print("说明：粗扫仅用于人工观察，不构成买入建议，也不会自动修改候选池或持仓。")


if __name__ == "__main__":
    main()
