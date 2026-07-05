from __future__ import annotations

from datetime import datetime

from src.data.cache_store import MarketDataCache
from src.data.providers.akshare_provider import AKShareProvider


def main() -> None:
    provider = AKShareProvider(cache=MarketDataCache(), use_cache=True)

    bundle = provider.get_bars(
        symbol="000001.SZ",
        start_time=datetime(2024, 1, 1),
        end_time=datetime(2024, 3, 1),
        frequency="1d",
        adjust_type="qfq",
    )

    print("provider:", bundle.provider)
    print("symbol:", bundle.symbol)
    print("rows:", len(bundle.bars))
    print("first:", bundle.bars[0] if bundle.bars else None)
    print("last:", bundle.bars[-1] if bundle.bars else None)


if __name__ == "__main__":
    main()
