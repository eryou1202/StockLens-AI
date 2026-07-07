from __future__ import annotations

import argparse

from src.reference.symbol_reference import AShareSymbolReference


def main() -> None:
    parser = argparse.ArgumentParser(description="搜索本地 A 股代码名称参考表")
    parser.add_argument("--query", required=True)
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--reference", default="data/reference/a_share_symbols.csv")
    args = parser.parse_args()

    reference = AShareSymbolReference(reference_path=args.reference)
    if not reference.reference_path.exists():
        print("参考表不存在，请先运行 python -m scripts.build_a_share_symbol_reference")
        return
    result = reference.search(args.query, args.limit)
    print(f"query: {args.query}")
    print(f"matches: {len(result)}")
    if result.empty:
        print("未找到匹配股票。")
    else:
        print(result.to_string(index=False))


if __name__ == "__main__":
    main()
