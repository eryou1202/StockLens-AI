from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd

from src.data.cache_store import MarketDataCache
from src.data.market_data_provider import MarketDataProvider
from src.data.symbol_mapper import SymbolMapper
from src.models.schemas import MarketBar, MarketDataBundle, StockStatus


class BaostockProvider(MarketDataProvider):
    """
    Baostock 行情数据源。

    安装：
        pip install baostock

    初版支持：
    - 日线 K 线
    - 5/15/30/60 分钟 K 线接口占位
    - 指数 K 线通过 get_index_bars 走 get_bars

    TODO:
    - 更严格的错误处理
    - 自动重试
    - query_trade_dates 交易日历
    - query_stock_basic 股票状态
    - 指数代码映射
    - 行业指数映射
    """

    provider_name = "baostock"

    # Baostock frequency:
    # d=日k线, w=周, m=月, 5=5分钟, 15=15分钟, 30=30分钟, 60=60分钟
    FREQ_MAP = {
        "1d": "d",
        "1w": "w",
        "1mo": "m",
        "5m": "5",
        "15m": "15",
        "30m": "30",
        "60m": "60",
    }

    # Baostock adjustflag:
    # 1=后复权, 2=前复权, 3=不复权
    ADJUST_MAP = {
        "hfq": "1",
        "qfq": "2",
        "none": "3",
    }

    DEFAULT_FIELDS = (
        "date,code,open,high,low,close,preclose,volume,amount,"
        "adjustflag,turn,tradestatus,pctChg,peTTM,pbMRQ,psTTM,pcfNcfTTM,isST"
    )

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
            import baostock as bs
        except ImportError as exc:
            raise ImportError("未安装 baostock，请运行：pip install baostock") from exc

        bs_symbol = SymbolMapper.to_baostock(internal_symbol)
        bs_frequency = self.FREQ_MAP.get(frequency)
        if not bs_frequency:
            raise ValueError(f"Baostock 暂不支持 frequency={frequency}")

        bs_adjust = self.ADJUST_MAP.get(adjust_type)
        if not bs_adjust:
            raise ValueError(f"Baostock 暂不支持 adjust_type={adjust_type}")

        lg = bs.login()
        if lg.error_code != "0":
            raise RuntimeError(f"Baostock login failed: {lg.error_msg}")

        try:
            rs = bs.query_history_k_data_plus(
                bs_symbol,
                self.DEFAULT_FIELDS,
                start_date=start_time.strftime("%Y-%m-%d"),
                end_date=end_time.strftime("%Y-%m-%d"),
                frequency=bs_frequency,
                adjustflag=bs_adjust,
            )

            if rs.error_code != "0":
                raise RuntimeError(f"Baostock query failed: {rs.error_msg}")

            rows: list[dict[str, Any]] = []
            while rs.next():
                rows.append(rs.get_row_data())

            df = pd.DataFrame(rows, columns=rs.fields)
            bars = self._frame_to_bars(
                df=df,
                internal_symbol=internal_symbol,
                frequency=frequency,
                adjust_type=adjust_type,
                raw_symbol=bs_symbol,
            )

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
                    "source_symbol": bs_symbol,
                },
            )

            if self.use_cache and self.cache:
                self.cache.save_bundle(bundle)

            return bundle
        finally:
            bs.logout()

    def get_trade_calendar(self, start_time: datetime, end_time: datetime) -> list[datetime]:
        """
        Baostock 交易日历。

        TODO:
        - 解析 is_trading_day
        - 只返回交易日
        """
        try:
            import baostock as bs
        except ImportError as exc:
            raise ImportError("未安装 baostock，请运行：pip install baostock") from exc

        lg = bs.login()
        if lg.error_code != "0":
            raise RuntimeError(f"Baostock login failed: {lg.error_msg}")

        try:
            rs = bs.query_trade_dates(
                start_date=start_time.strftime("%Y-%m-%d"),
                end_date=end_time.strftime("%Y-%m-%d"),
            )
            if rs.error_code != "0":
                raise RuntimeError(f"Baostock query_trade_dates failed: {rs.error_msg}")

            days = []
            while rs.next():
                row = dict(zip(rs.fields, rs.get_row_data()))
                if row.get("is_trading_day") == "1":
                    days.append(datetime.strptime(row["calendar_date"], "%Y-%m-%d"))
            return days
        finally:
            bs.logout()

    def get_stock_status(self, symbol: str, as_of_time: datetime) -> StockStatus:
        """
        查询股票状态。

        TODO:
        - query_stock_basic 结果字段细节需要根据实际返回再完善
        - 与 K 线中的 tradestatus / isST 合并
        """
        internal_symbol = SymbolMapper.normalize(symbol)
        bs_symbol = SymbolMapper.to_baostock(internal_symbol)

        try:
            import baostock as bs
        except ImportError as exc:
            raise ImportError("未安装 baostock，请运行：pip install baostock") from exc

        lg = bs.login()
        if lg.error_code != "0":
            raise RuntimeError(f"Baostock login failed: {lg.error_msg}")

        try:
            rs = bs.query_stock_basic(code=bs_symbol)
            raw = {}
            if rs.error_code == "0" and rs.next():
                raw = dict(zip(rs.fields, rs.get_row_data()))

            return StockStatus(
                symbol=internal_symbol,
                as_of_time=as_of_time,
                is_trading=None,
                is_st=None,
                listed_date=self._parse_date_or_none(raw.get("ipoDate")),
                delisted_date=self._parse_date_or_none(raw.get("outDate")),
                provider=self.provider_name,
                raw=raw,
            )
        finally:
            bs.logout()

    def _frame_to_bars(
        self,
        df: pd.DataFrame,
        internal_symbol: str,
        frequency: str,
        adjust_type: str,
        raw_symbol: str,
    ) -> list[MarketBar]:
        bars: list[MarketBar] = []
        fetched_at = datetime.now()

        for _, row in df.iterrows():
            raw = row.to_dict()
            trade_time = self._parse_datetime(str(raw.get("date")))

            bars.append(
                MarketBar(
                    symbol=internal_symbol,
                    trade_time=trade_time,
                    frequency=frequency,
                    adjust_type=adjust_type,
                    open=self._to_float(raw.get("open")),
                    high=self._to_float(raw.get("high")),
                    low=self._to_float(raw.get("low")),
                    close=self._to_float(raw.get("close")),
                    pre_close=self._to_float(raw.get("preclose")),
                    volume=self._to_float(raw.get("volume")),
                    amount=self._to_float(raw.get("amount")),
                    turnover_rate=self._to_float(raw.get("turn")),
                    pct_chg=self._to_float(raw.get("pctChg")),
                    trade_status=self._to_str_or_none(raw.get("tradestatus")),
                    is_st=self._to_bool_st(raw.get("isST")),
                    pe_ttm=self._to_float(raw.get("peTTM")),
                    pb=self._to_float(raw.get("pbMRQ")),
                    ps_ttm=self._to_float(raw.get("psTTM")),
                    pcf_ncf_ttm=self._to_float(raw.get("pcfNcfTTM")),
                    provider=self.provider_name,
                    raw_symbol=raw_symbol,
                    fetched_at=fetched_at,
                    raw=raw,
                )
            )

        return bars

    @staticmethod
    def _parse_datetime(value: str) -> datetime:
        # Baostock 日线一般是 YYYY-MM-DD。
        # 分钟线如果返回更细时间，后续在这里扩展。
        return pd.to_datetime(value).to_pydatetime()

    @staticmethod
    def _parse_date_or_none(value: Any) -> datetime | None:
        if value in (None, "", "None"):
            return None
        return pd.to_datetime(value).to_pydatetime()

    @staticmethod
    def _to_float(value: Any) -> float | None:
        if value in (None, "", "None"):
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _to_str_or_none(value: Any) -> str | None:
        if value in (None, "", "None"):
            return None
        return str(value)

    @staticmethod
    def _to_bool_st(value: Any) -> bool | None:
        if value in (None, "", "None"):
            return None
        return str(value) == "1"
