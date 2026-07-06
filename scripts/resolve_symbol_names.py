from __future__ import annotations

from src.config.settings import load_settings
from src.data.symbol_name_resolver import SymbolNameResolver


def main() -> None:
    settings = load_settings()
    resolver = SymbolNameResolver(settings.database_path, "data/ai_candidates.json")
    result = resolver.backfill_all()
    print("StockLens 股票名称补全")
    print(f"扫描数量: {result['total']}")
    print(f"补全成功数量: {result['success']}")
    print(f"补全失败数量: {result['failed']}")
    if result["failed_symbols"]:
        print("失败 symbol: " + ", ".join(result["failed_symbols"]))
    else:
        print("失败 symbol: 无")


if __name__ == "__main__":
    main()
