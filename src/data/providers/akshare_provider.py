from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd

from src.data.cache_store import MarketDataCache
from src.data.market_data_provider import MarketDataProvider
from src.data.symbol_mapper import SymbolMapper
from src.models.schemas import MarketBar, MarketDataBundle, StockStatus


class AKShareProvider(MarketDataProvider):
    """
    AKShare 行情数据源。

    安装：
        pip install akshare

    初版支持：
    - stock_zh_a_hist 日线/周线/月线
    - stock_zh_a_minute 分钟线接口占位
    - 指数和行业接口 TODO

    TODO:
    - AKShare 不同接口字段名可能变化，要加字段兼容层
    - 指数行情使用 stock_zh_index_daily / stock_zh_index_daily_em 等接口
    - 分钟数据接口与日线接口返回字段可能不同
    """

    provider_name = "akshare"

    PERIOD_MAP = {
        "1d": "daily",
        "1w": "weekly",
        "1mo": "monthly",
    }

    MINUTE_PERIOD_MAP = {
        "1m": "1",
        "5m": "5",
        "15m": "15",
        "30m": "30",
        "60m": "60",
    }

    ADJUST_MAP = {
        "none": "",
        "qfq": "qfq",
        "hfq": "hfq",
    }

    def __init__(self, cache: MarketDataCache | None = None, use_cache: bool = True):
        self.cache = cache
        self.use_cache = use_cache

    def get_bars(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
        frequency: str = "1d",
        adjust_type: str = "qfq",
    ) -> MarketDataBundle:
        internal_symbol = SymbolMapper.normalize(symbol)

        if self.use_cache and self.cache:
            cached = self.cache.load_bundle(
                provider=self.provider_name,
                symbol=internal_symbol,
                start_time=start_time,
                end_time=end_time,
                frequency=frequency,
                adjust_type=adjust_type,
            )
            if cached:
                return cached

        try:
            import akshare as ak
        except ImportError as exc:
            raise ImportError("未安装 akshare，请运行：pip install akshare") from exc

        ak_symbol = SymbolMapper.to_akshare(internal_symbol)
        adjust = self.ADJUST_MAP.get(adjust_type)
        if adjust is None:
            raise ValueError(f"AKShare 暂不支持 adjust_type={adjust_type}")

        if frequency in self.PERIOD_MAP:
            df = ak.stock_zh_a_hist(
                symbol=ak_symbol,
                period=self.PERIOD_MAP[frequency],
                start_date=start_time.strftime("%Y%m%d"),
                end_date=end_time.strftime("%Y%m%d"),
                adjust=adjust,
            )
        elif frequency in self.MINUTE_PERIOD_MAP:
            # TODO:
            # AKShare 分钟接口 stock_zh_a_minute 的可用参数和返回范围需要实测。
            # 有些 period / adjust 组合可能不支持。
            df = ak.stock_zh_a_minute(
                symbol=ak_symbol,
                period=self.MINUTE_PERIOD_MAP[frequency],
                adjust=adjust,
            )
            # 分钟接口可能需要手动按时间切片。
        else:
            raise ValueError(f"AKShare 暂不支持 frequency={frequency}")

        bars = self._frame_to_bars(
            df=df,
            internal_symbol=internal_symbol,
            frequency=frequency,
            adjust_type=adjust_type,
            raw_symbol=ak_symbol,
        )

        # 保险：按请求时间过滤
        bars = [bar for bar in bars if start_time <= bar.trade_time <= end_time]

        bundle = MarketDataBundle(
            symbol=internal_symbol,
            start_time=start_time,
            end_time=end_time,
            frequency=frequency,
            adjust_type=adjust_type,
            provider=self.provider_name,
            bars=bars,
            data_quality={
                "from_cache": False,
                "rows": len(bars),
                "source_symbol": ak_symbol,
            },
        )

        if self.use_cache and self.cache:
            self.cache.save_bundle(bundle)

        return bundle

    def get_index_bars(
        self,
        index_symbol: str,
        start_time: datetime,
        end_time: datetime,
        frequency: str = "1d",
        adjust_type: str = "none",
    ) -> MarketDataBundle:
        """
        AKShare 指数行情接口占位。

        TODO:
        - 使用 ak.stock_zh_index_daily 或 ak.stock_zh_index_daily_em
        - 做指数代码映射：000300.SH -> sh000300 / 000300 等
        """
        raise NotImplementedError("AKShareProvider.get_index_bars 需要按指数接口进一步实现。")

    def get_trade_calendar(self, start_time: datetime, end_time: datetime) -> list[datetime]:
        """
        AKShare 交易日历接口占位。

        TODO:
        - 可考虑 ak.tool_trade_date_hist_sina()
        - 当前先用父类的工作日近似
        """
        return super().get_trade_calendar(start_time, end_time)

    def get_stock_status(self, symbol: str, as_of_time: datetime) -> StockStatus:
        """
        AKShare 股票状态接口占位。

        TODO:
        - stock_individual_info_em
        - 实时行情里解析是否停牌
        - ST 状态可从名称或专项接口判断
        """
        return StockStatus(
            symbol=SymbolMapper.normalize(symbol),
            as_of_time=as_of_time,
            is_trading=None,
            is_st=None,
            provider=self.provider_name,
        )

    def _frame_to_bars(
        self,
        df: pd.DataFrame,
        internal_symbol: str,
        frequency: str,
        adjust_type: str,
        raw_symbol: str,
    ) -> list[MarketBar]:
        """
        将 AKShare 返回的中文字段 DataFrame 转换为 MarketBar。

        常见 stock_zh_a_hist 字段：
        日期, 股票代码, 开盘, 收盘, 最高, 最低, 成交量, 成交额, 振幅, 涨跌幅, 涨跌额, 换手率
        """
        if df is None or df.empty:
            return []

        fetched_at = datetime.now()
        bars: list[MarketBar] = []

        for _, row in df.iterrows():
            raw = row.to_dict()

            trade_time = self._get_datetime(raw, ["日期", "时间", "day", "date"])
            if trade_time is None:
                continue

            bars.append(
                MarketBar(
                    symbol=internal_symbol,
                    trade_time=trade_time,
                    frequency=frequency,
                    adjust_type=adjust_type,
                    open=self._get_float(raw, ["开盘", "open"]),
                    high=self._get_float(raw, ["最高", "high"]),
                    low=self._get_float(raw, ["最低", "low"]),
                    close=self._get_float(raw, ["收盘", "close"]),
                    pre_close=None,  # AKShare stock_zh_a_hist 通常不直接给昨收
                    volume=self._get_float(raw, ["成交量", "volume"]),
                    amount=self._get_float(raw, ["成交额", "amount"]),
                    turnover_rate=self._get_float(raw, ["换手率", "turnover"]),
                    pct_chg=self._get_float(raw, ["涨跌幅", "pct_chg", "pctChg"]),
                    trade_status=None,
                    is_st=None,
                    provider=self.provider_name,
                    raw_symbol=raw_symbol,
                    fetched_at=fetched_at,
                    raw=raw,
                )
            )

        return bars

    @staticmethod
    def _get_float(raw: dict[str, Any], keys: list[str]) -> float | None:
        for key in keys:
            if key in raw and raw[key] not in (None, "", "None"):
                try:
                    return float(raw[key])
                except (ValueError, TypeError):
                    return None
        return None

    @staticmethod
    def _get_datetime(raw: dict[str, Any], keys: list[str]) -> datetime | None:
        for key in keys:
            if key in raw and raw[key] not in (None, "", "None"):
                try:
                    return pd.to_datetime(raw[key]).to_pydatetime()
                except Exception:
                    return None
        return None
