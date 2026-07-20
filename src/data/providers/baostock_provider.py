from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
from typing import Any, Iterator

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
        self._session_depth = 0
        self._logged_in = False
        self._bs: Any | None = None
        self._last_logout_error: str | None = None
        self._stats = {
            "login_count": 0,
            "logout_count": 0,
            "history_query_count": 0,
            "trade_calendar_query_count": 0,
            "stock_basic_query_count": 0,
            "all_stock_query_count": 0,
            "cache_hit_count": 0,
        }

    @contextmanager
    def session(self) -> Iterator[BaostockProvider]:
        # Baostock uses process-global login state. This session is for
        # sequential work in the current Python process/thread only.
        self._session_depth += 1
        try:
            yield self
        finally:
            self._session_depth = max(0, self._session_depth - 1)
            if self._session_depth == 0:
                self.close()

    def close(self) -> None:
        if not self._logged_in or self._bs is None:
            self._logged_in = False
            self._bs = None
            return
        try:
            self._bs.logout()
            self._stats["logout_count"] += 1
            self._last_logout_error = None
        except Exception as exc:
            self._last_logout_error = f"{type(exc).__name__}: {exc}"
        finally:
            self._logged_in = False
            self._bs = None

    def session_stats(self) -> dict[str, int]:
        return dict(self._stats)

    def reset_session_stats(self) -> None:
        for key in self._stats:
            self._stats[key] = 0
        self._last_logout_error = None

    def _ensure_login(self) -> Any:
        if self._logged_in and self._bs is not None:
            return self._bs
        try:
            import baostock as bs
        except ImportError as exc:
            raise ImportError("baostock is not installed; run: pip install baostock") from exc
        lg = bs.login()
        if lg.error_code != "0":
            self._logged_in = False
            self._bs = None
            raise RuntimeError(f"Baostock login failed: {lg.error_msg}")
        self._bs = bs
        self._logged_in = True
        self._stats["login_count"] += 1
        return bs

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
                self._stats["cache_hit_count"] += 1
                return cached

        bs_frequency = self.FREQ_MAP.get(frequency)
        if not bs_frequency:
            raise ValueError(f"Baostock 暂不支持 frequency={frequency}")

        bs_adjust = self.ADJUST_MAP.get(adjust_type)
        if not bs_adjust:
            raise ValueError(f"Baostock 暂不支持 adjust_type={adjust_type}")

        with self.session():
            bundle = self.query_history_bars(
                symbol=internal_symbol,
                start_time=start_time,
                end_time=end_time,
                frequency=frequency,
                adjust_type=adjust_type,
            )

        if self.use_cache and self.cache:
            self.cache.save_bundle(bundle)

        return bundle

    def query_history_bars(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
        frequency: str = "1d",
        adjust_type: str = "qfq",
    ) -> MarketDataBundle:
        if self._session_depth == 0:
            with self.session():
                return self.query_history_bars(symbol, start_time, end_time, frequency, adjust_type)
        internal_symbol = SymbolMapper.normalize(symbol)
        bs_symbol = SymbolMapper.to_baostock(internal_symbol)
        bs_frequency = self.FREQ_MAP.get(frequency)
        if not bs_frequency:
            raise ValueError(f"Baostock 暂不支持 frequency={frequency}")
        bs_adjust = self.ADJUST_MAP.get(adjust_type)
        if not bs_adjust:
            raise ValueError(f"Baostock 暂不支持 adjust_type={adjust_type}")

        bs = self._ensure_login()
        self._stats["history_query_count"] += 1
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

        return MarketDataBundle(
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

    def get_trade_calendar(self, start_time: datetime, end_time: datetime) -> list[datetime]:
        """
        Baostock 交易日历。

        TODO:
        - 解析 is_trading_day
        - 只返回交易日
        """
        with self.session():
            return self.query_trade_calendar(start_time, end_time)

    def query_trade_calendar(self, start_time: datetime, end_time: datetime) -> list[datetime]:
        if self._session_depth == 0:
            with self.session():
                return self.query_trade_calendar(start_time, end_time)
        bs = self._ensure_login()
        self._stats["trade_calendar_query_count"] += 1
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

    def get_stock_status(self, symbol: str, as_of_time: datetime) -> StockStatus:
        """
        查询股票状态。

        TODO:
        - query_stock_basic 结果字段细节需要根据实际返回再完善
        - 与 K 线中的 tradestatus / isST 合并
        """
        internal_symbol = SymbolMapper.normalize(symbol)

        with self.session():
            raw = self.query_stock_basic(internal_symbol)

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

    def query_stock_basic(self, symbol: str) -> dict[str, Any]:
        if self._session_depth == 0:
            with self.session():
                return self.query_stock_basic(symbol)
        internal_symbol = SymbolMapper.normalize(symbol)
        bs_symbol = SymbolMapper.to_baostock(internal_symbol)
        bs = self._ensure_login()
        self._stats["stock_basic_query_count"] += 1
        rs = bs.query_stock_basic(code=bs_symbol)
        raw: dict[str, Any] = {}
        if rs.error_code == "0" and rs.next():
            raw = dict(zip(rs.fields, rs.get_row_data()))
        return raw

    def get_stock_name(self, symbol: str) -> str | None:
        with self.session():
            raw = self.query_stock_basic(symbol)
        return self.stock_name_from_basic(raw)

    def query_all_stock(self, query_date: datetime) -> list[dict[str, Any]]:
        if self._session_depth == 0:
            with self.session():
                return self.query_all_stock(query_date)
        bs = self._ensure_login()
        self._stats["all_stock_query_count"] += 1
        result = bs.query_all_stock(query_date.strftime("%Y-%m-%d"))
        if result.error_code != "0":
            raise RuntimeError(f"Baostock query_all_stock failed: {result.error_msg}")
        rows: list[dict[str, Any]] = []
        while result.next():
            rows.append(dict(zip(result.fields, result.get_row_data())))
        return rows

    @staticmethod
    def stock_name_from_basic(raw: dict[str, Any]) -> str | None:
        for key in ("code_name", "codeName", "name"):
            value = raw.get(key)
            if value:
                text = str(value).strip()
                if text:
                    return text
        return None

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
