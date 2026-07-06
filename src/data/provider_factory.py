from __future__ import annotations

from src.data.cache_store import MarketDataCache
from src.data.market_data_provider import MarketDataProvider, MockMarketDataProvider
from src.data.providers.akshare_provider import AKShareProvider
from src.data.providers.baostock_provider import BaostockProvider


def create_market_data_provider(
    name: str,
    cache_dir: str = "data/cache",
    use_cache: bool = True,
) -> MarketDataProvider:
    """
    根据配置创建行情数据源。

    name:
    - mock
    - baostock
    - akshare
    """
    name = name.lower().strip()
    cache = MarketDataCache(cache_dir)

    if name == "mock":
        return MockMarketDataProvider()
    if name == "baostock":
        return BaostockProvider(cache=cache, use_cache=use_cache)
    if name == "akshare":
        return AKShareProvider(cache=cache, use_cache=use_cache)

    raise ValueError(f"未知 market data provider: {name}")
