from __future__ import annotations

import math
import threading
from datetime import datetime, timedelta
from typing import Any

import pandas as pd
from pydantic import BaseModel

from src.data.symbol_mapper import SymbolMapper
from src.models.schemas import MarketBar


class RealtimeQuote(BaseModel):
    symbol: str
    stock_name: str | None = None
    quote_time: datetime
    latest_price: float | None = None
    prev_close: float | None = None
    open: float | None = None
    high: float | None = None
    low: float | None = None
    volume: float | None = None
    amount: float | None = None
    turnover_rate: float | None = None
    pct_change: float | None = None
    change_amount: float | None = None
    amplitude: float | None = None
    volume_ratio: float | None = None
    source: str = "akshare_eastmoney"


class CurrentPriceSnapshot(BaseModel):
    symbol: str
    current_price: float | None = None
    price_time: datetime | None = None
    price_source: str = "unavailable"
    is_realtime: bool = False
    is_stale: bool = True
    realtime_pct_change: float | None = None
    realtime_source: str | None = None
    quote: RealtimeQuote | None = None
    warning: str | None = None


class RealtimeQuoteProvider:
    """Short-lived A-share quote overlay. It never writes to historical bars."""

    _cache_lock = threading.Lock()
    _cached_at: datetime | None = None
    _cached_quotes: dict[str, RealtimeQuote] = {}

    def __init__(self, cache_seconds: int = 30) -> None:
        self.cache_seconds = max(0, int(cache_seconds))
        self.last_warning: str | None = None

    def get_all_quotes(self, force_refresh: bool = False) -> dict[str, RealtimeQuote]:
        now = datetime.now()
        with self._cache_lock:
            if (
                not force_refresh
                and self._cached_at is not None
                and now - self._cached_at <= timedelta(seconds=self.cache_seconds)
            ):
                self.last_warning = None
                return dict(self._cached_quotes)

        try:
            import akshare as ak
        except ImportError:
            self.last_warning = "未安装 akshare，当前只能使用日线数据。请运行：pip install akshare"
            return {}

        try:
            frame = ak.stock_zh_a_spot_em()
            quotes = self._frame_to_quotes(frame, now)
        except Exception as exc:
            self.last_warning = f"实时行情不可用，未做盘中确认：{type(exc).__name__}: {exc}"
            return {}
        if not quotes:
            self.last_warning = "实时行情接口未返回有效 A 股报价，当前只能使用日线数据。"
            return {}

        with self._cache_lock:
            type(self)._cached_at = now
            type(self)._cached_quotes = dict(quotes)
        self.last_warning = None
        return quotes

    def get_quote(
        self,
        symbol: str,
        force_refresh: bool = False,
        quotes: dict[str, RealtimeQuote] | None = None,
    ) -> RealtimeQuote | None:
        normalized = SymbolMapper.normalize(symbol)
        values = self.get_all_quotes(force_refresh=force_refresh) if quotes is None else quotes
        return values.get(normalized)

    def get_current_price(
        self,
        symbol: str,
        latest_market_bar: MarketBar | None = None,
        force_refresh: bool = False,
        quotes: dict[str, RealtimeQuote] | None = None,
    ) -> CurrentPriceSnapshot:
        normalized = SymbolMapper.normalize(symbol)
        quote = self.get_quote(normalized, force_refresh=force_refresh, quotes=quotes)
        if quote is not None:
            latest = self._number(quote.latest_price)
            if latest is not None and latest > 0:
                return CurrentPriceSnapshot(
                    symbol=normalized,
                    current_price=latest,
                    price_time=quote.quote_time,
                    price_source="realtime_quote",
                    is_realtime=True,
                    is_stale=(datetime.now() - quote.quote_time) > timedelta(seconds=90),
                    realtime_pct_change=quote.pct_change,
                    realtime_source=quote.source,
                    quote=quote,
                    warning=self.last_warning,
                )

        if latest_market_bar is not None:
            close = self._number(latest_market_bar.close)
            if close is not None and close > 0:
                trade_time = latest_market_bar.trade_time
                return CurrentPriceSnapshot(
                    symbol=normalized,
                    current_price=close,
                    price_time=trade_time,
                    price_source="latest_market_bar",
                    is_realtime=False,
                    is_stale=trade_time.date() < datetime.now().date(),
                    realtime_source=None,
                    warning=self.last_warning or "实时行情不可用，未做盘中确认。",
                )

        return CurrentPriceSnapshot(
            symbol=normalized,
            price_source="unavailable",
            is_realtime=False,
            is_stale=True,
            warning=self.last_warning or "实时行情与最新日线均不可用。",
        )

    @classmethod
    def _frame_to_quotes(
        cls,
        frame: pd.DataFrame,
        quote_time: datetime,
    ) -> dict[str, RealtimeQuote]:
        if frame is None or frame.empty:
            return {}
        quotes: dict[str, RealtimeQuote] = {}
        for row in frame.to_dict("records"):
            symbol = cls._normalize_code(row.get("代码") or row.get("code"))
            if symbol is None:
                continue
            quotes[symbol] = RealtimeQuote(
                symbol=symbol,
                stock_name=cls._text(row.get("名称") or row.get("name")),
                quote_time=quote_time,
                latest_price=cls._number(row.get("最新价")),
                prev_close=cls._number(row.get("昨收")),
                open=cls._number(row.get("今开")),
                high=cls._number(row.get("最高")),
                low=cls._number(row.get("最低")),
                volume=cls._number(row.get("成交量")),
                amount=cls._number(row.get("成交额")),
                turnover_rate=cls._percent(row.get("换手率")),
                pct_change=cls._percent(row.get("涨跌幅")),
                change_amount=cls._number(row.get("涨跌额")),
                amplitude=cls._percent(row.get("振幅")),
                volume_ratio=cls._number(row.get("量比")),
            )
        return quotes

    @staticmethod
    def _normalize_code(value: Any) -> str | None:
        code = str(value or "").strip().zfill(6)
        if len(code) != 6 or not code.isdigit():
            return None
        if code.startswith(("600", "601", "603", "605", "688", "689")):
            return f"{code}.SH"
        if code.startswith(("000", "001", "002", "003", "300", "301")):
            return f"{code}.SZ"
        if code.startswith(("4", "8", "92")):
            return f"{code}.BJ"
        return None

    @staticmethod
    def _number(value: Any) -> float | None:
        if value is None:
            return None
        try:
            result = float(str(value).replace(",", ""))
        except (TypeError, ValueError, OverflowError):
            return None
        return result if math.isfinite(result) else None

    @classmethod
    def _percent(cls, value: Any) -> float | None:
        result = cls._number(value)
        return None if result is None else result / 100.0

    @staticmethod
    def _text(value: Any) -> str | None:
        if value is None or pd.isna(value):
            return None
        text = str(value).strip()
        return text or None


def get_current_price(
    symbol: str,
    latest_market_bar: MarketBar | None = None,
    force_refresh: bool = False,
    quotes: dict[str, RealtimeQuote] | None = None,
    provider: RealtimeQuoteProvider | None = None,
) -> CurrentPriceSnapshot:
    return (provider or RealtimeQuoteProvider()).get_current_price(
        symbol=symbol,
        latest_market_bar=latest_market_bar,
        force_refresh=force_refresh,
        quotes=quotes,
    )
