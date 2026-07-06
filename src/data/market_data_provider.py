from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timedelta

import numpy as np

from src.models.schemas import MarketBar, MarketDataBundle, StockStatus


class MarketDataProvider(ABC):
    """
    行情数据统一接口。

    QuantEngine 只依赖这个接口，不直接依赖 baostock / akshare。
    """

    provider_name: str = "abstract"

    @abstractmethod
    def get_bars(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
        frequency: str = "1d",
        adjust_type: str = "qfq",
    ) -> MarketDataBundle:
        """
        获取股票历史行情曲线。

        frequency:
        - 1d
        - 5m
        - 15m
        - 30m
        - 60m

        adjust_type:
        - none: 不复权
        - qfq: 前复权
        - hfq: 后复权
        """
        raise NotImplementedError

    def get_latest_quote(
        self,
        symbol: str,
        as_of_time: datetime,
        frequency: str = "1d",
        adjust_type: str = "qfq",
    ) -> MarketBar | None:
        """
        获取当前或最近一个可用行情。

        TODO:
        - 真实行情需要考虑盘中、收盘、周末、节假日。
        """
        bundle = self.get_bars(
            symbol=symbol,
            start_time=as_of_time - timedelta(days=10),
            end_time=as_of_time,
            frequency=frequency,
            adjust_type=adjust_type,
        )
        bars = bundle.sorted_bars()
        return bars[-1] if bars else None

    def get_index_bars(
        self,
        index_symbol: str,
        start_time: datetime,
        end_time: datetime,
        frequency: str = "1d",
        adjust_type: str = "none",
    ) -> MarketDataBundle:
        """
        获取指数行情。

        TODO:
        - 统一指数代码格式
        - Baostock 对指数可用 sh.000300 等
        - AKShare 需要单独 index 接口，后续完善
        """
        return self.get_bars(index_symbol, start_time, end_time, frequency, adjust_type)

    def get_industry_bars(
        self,
        stock_symbol: str,
        start_time: datetime,
        end_time: datetime,
        frequency: str = "1d",
        adjust_type: str = "none",
    ) -> MarketDataBundle | None:
        """
        获取股票对应行业指数。

        TODO:
        - 建立 stock -> industry 映射
        - 接入申万 / 中证行业指数
        """
        return None

    def get_trade_calendar(self, start_time: datetime, end_time: datetime) -> list[datetime]:
        """
        获取交易日历。

        TODO:
        - Baostock: query_trade_dates
        - AKShare: tool_trade_date_hist_sina 或相关接口
        """
        days = []
        current = start_time
        while current <= end_time:
            if current.weekday() < 5:
                days.append(current)
            current += timedelta(days=1)
        return days

    def get_stock_status(self, symbol: str, as_of_time: datetime) -> StockStatus:
        """
        获取股票状态。

        TODO:
        - 是否停牌
        - 是否 ST
        - 是否上市/退市
        """
        return StockStatus(
            symbol=symbol,
            as_of_time=as_of_time,
            is_trading=None,
            is_st=None,
            provider=self.provider_name,
        )


class MockMarketDataProvider(MarketDataProvider):
    """
    Mock 数据源，生成随机行情。

    用于没装 baostock / akshare 时跑通流程。
    """

    provider_name = "mock"

    def get_bars(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
        frequency: str = "1d",
        adjust_type: str = "qfq",
    ) -> MarketDataBundle:
        rng = np.random.default_rng(abs(hash(symbol)) % (2**32))

        days = max((end_time.date() - start_time.date()).days, 30)
        dates = [start_time + timedelta(days=i) for i in range(days + 1)]
        dates = [d for d in dates if d.weekday() < 5]

        base = 10 + (abs(hash(symbol)) % 2000) / 100
        returns = rng.normal(loc=0.0005, scale=0.018, size=len(dates))
        prices = base * np.cumprod(1 + returns)

        bars: list[MarketBar] = []
        prev_close = None
        for dt, close in zip(dates, prices):
            open_price = float(close * (1 + rng.normal(0, 0.004)))
            high = float(max(open_price, close) * (1 + abs(rng.normal(0, 0.008))))
            low = float(min(open_price, close) * (1 - abs(rng.normal(0, 0.008))))
            volume = float(rng.integers(1_000_000, 20_000_000))
            pct_chg = None
            if prev_close:
                pct_chg = (float(close) / prev_close - 1) * 100

            bars.append(
                MarketBar(
                    symbol=symbol,
                    trade_time=dt,
                    frequency=frequency,
                    adjust_type=adjust_type,
                    open=open_price,
                    high=high,
                    low=low,
                    close=float(close),
                    pre_close=prev_close,
                    volume=volume,
                    amount=float(volume * close),
                    turnover_rate=float(rng.uniform(0.2, 5.0)),
                    pct_chg=pct_chg,
                    trade_status="1",
                    is_st=False,
                    provider=self.provider_name,
                    raw_symbol=symbol,
                )
            )
            prev_close = float(close)

        return MarketDataBundle(
            symbol=symbol,
            start_time=start_time,
            end_time=end_time,
            frequency=frequency,
            adjust_type=adjust_type,
            provider=self.provider_name,
            bars=bars,
            data_quality={
                "from_cache": False,
                "rows": len(bars),
                "mock": True,
            },
        )
