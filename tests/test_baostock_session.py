from __future__ import annotations

import sqlite3
import sys
import tempfile
import unittest
import gc
from contextlib import contextmanager
from datetime import datetime, time, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

sys.modules.setdefault("yaml", SimpleNamespace(safe_load=lambda _text: {}))

from src.data.market_data_provider import MarketDataProvider
from src.data.providers.baostock_provider import BaostockProvider
from src.data.symbol_name_resolver import SymbolNameResolver
from src.feedback.feedback_engine import FeedbackEngine
from src.models.schemas import MarketBar, MarketDataBundle
from src.scan.a_share_coarse_scanner import AShareCoarseScanner
from src.storage.sqlite_store import SQLiteSignalStore
from src.tracking.recommendation_tracker import RecommendationTracker


class FakeResult:
    def __init__(self, fields: list[str], rows: list[list[str]], error_code: str = "0"):
        self.fields = fields
        self.rows = rows
        self.error_code = error_code
        self.error_msg = "fake error" if error_code != "0" else ""
        self._index = -1

    def next(self) -> bool:
        self._index += 1
        return self._index < len(self.rows)

    def get_row_data(self) -> list[str]:
        return self.rows[self._index]


class FakeBaostock:
    def __init__(self, fail_history_call: int | None = None):
        self.login_count = 0
        self.logout_count = 0
        self.history_query_count = 0
        self.trade_calendar_query_count = 0
        self.stock_basic_query_count = 0
        self.all_stock_query_count = 0
        self.fail_history_call = fail_history_call

    def login(self):
        self.login_count += 1
        return SimpleNamespace(error_code="0", error_msg="")

    def logout(self):
        self.logout_count += 1

    def query_history_k_data_plus(
        self,
        code,
        fields,
        start_date,
        end_date,
        frequency,
        adjustflag,
    ):
        self.history_query_count += 1
        if self.fail_history_call == self.history_query_count:
            raise ValueError("history boom")
        field_list = fields.split(",")
        rows = []
        current = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        index = 0
        while current <= end:
            close = 10.0 + index * 0.1
            raw = {
                "date": current.strftime("%Y-%m-%d"),
                "code": code,
                "open": f"{close:.2f}",
                "high": f"{close + 0.2:.2f}",
                "low": f"{close - 0.2:.2f}",
                "close": f"{close:.2f}",
                "preclose": f"{max(1.0, close - 0.1):.2f}",
                "volume": "1000000",
                "amount": "10000000",
                "adjustflag": adjustflag,
                "turn": "1.0",
                "tradestatus": "1",
                "pctChg": "1.0",
                "peTTM": "10",
                "pbMRQ": "1",
                "psTTM": "1",
                "pcfNcfTTM": "1",
                "isST": "0",
            }
            rows.append([raw.get(field, "") for field in field_list])
            current += timedelta(days=1)
            index += 1
        return FakeResult(field_list, rows)

    def query_trade_dates(self, start_date, end_date):
        self.trade_calendar_query_count += 1
        return FakeResult(
            ["calendar_date", "is_trading_day"],
            [[start_date, "1"], [end_date, "1"]],
        )

    def query_stock_basic(self, code):
        self.stock_basic_query_count += 1
        suffix = str(code).split(".")[-1]
        return FakeResult(
            ["code", "code_name", "ipoDate", "outDate"],
            [[code, f"Name {suffix}", "2020-01-01", ""]],
        )

    def query_all_stock(self, date):
        self.all_stock_query_count += 1
        return FakeResult(
            ["code", "code_name", "tradeStatus"],
            [
                ["sz.000001", "Name 000001", "1"],
                ["sz.000002", "Name 000002", "1"],
                ["sh.600000", "Name 600000", "1"],
            ],
        )


@contextmanager
def fake_baostock(fake: FakeBaostock):
    with patch.dict(sys.modules, {"baostock": fake}):
        yield fake


def provider_stats(provider: BaostockProvider) -> dict[str, int]:
    return provider.session_stats()


def make_bundle(
    symbol: str,
    start_time: datetime,
    end_time: datetime,
    provider: str = "fake",
) -> MarketDataBundle:
    bars = []
    current = datetime.combine(start_time.replace(tzinfo=None).date(), time(15, 0))
    end = datetime.combine(end_time.replace(tzinfo=None).date(), time(15, 0))
    index = 0
    while current <= end:
        close = 10.0 + index * 0.1
        bars.append(
            MarketBar(
                symbol=symbol,
                trade_time=current,
                open=close,
                high=close + 0.2,
                low=close - 0.2,
                close=close,
                pre_close=max(1.0, close - 0.1),
                volume=1_000_000,
                amount=10_000_000,
                trade_status="1",
                provider=provider,
            )
        )
        current += timedelta(days=1)
        index += 1
    return MarketDataBundle(
        symbol=symbol,
        start_time=start_time,
        end_time=end_time,
        provider=provider,
        bars=bars,
    )


class AlwaysHitCache:
    def load_bundle(self, provider, symbol, start_time, end_time, frequency, adjust_type):
        return make_bundle(symbol, start_time, end_time, provider=provider)

    def save_bundle(self, bundle):
        raise AssertionError("cache hit should not save")


class CountingProvider(MarketDataProvider):
    provider_name = "counting"

    def __init__(self):
        self.history_query_count = 0
        self.queries: list[tuple[str, datetime, datetime]] = []

    def get_bars(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
        frequency: str = "1d",
        adjust_type: str = "qfq",
    ) -> MarketDataBundle:
        self.history_query_count += 1
        self.queries.append((symbol, start_time, end_time))
        return make_bundle(symbol, start_time, end_time, provider=self.provider_name)


class FakeFeedbackStore:
    def __init__(self, signals):
        self.signals = signals
        self.upserts = []

    def list_signals_for_feedback(self, limit=None):
        return self.signals[:limit] if limit is not None else list(self.signals)

    def upsert_feedback(self, signal_id, label, metadata, as_of_time_key=None):
        self.upserts.append((signal_id, label, metadata, as_of_time_key))


def insert_tracking_row(connection, symbol: str, as_of: datetime) -> None:
    now = datetime.now().isoformat()
    connection.execute(
        """
        INSERT INTO recommendation_tracking (
            symbol, stock_name, as_of_time, source_type, action, action_level,
            confidence, final_score, ai_view, quant_decision, final_level,
            current_price, suggested_horizon_days_json, reason_json, risks_json,
            invalid_conditions_json, tracking_status, created_at, updated_at, metadata_json
        ) VALUES (?, ?, ?, 'test', 'watch', 'watch', 0.5, 60.0, 'neutral',
                  'support', 'watch', 10.0, '[]', '[]', '[]', '[]',
                  'tracking', ?, ?, '{}')
        """,
        (symbol, symbol, as_of.isoformat(), now, now),
    )


class BaostockSessionTests(unittest.TestCase):
    def test_single_get_bars_uses_one_temporary_session(self):
        with fake_baostock(FakeBaostock()):
            provider = BaostockProvider(cache=None, use_cache=False)
            provider.get_bars("000001.SZ", datetime(2026, 1, 1), datetime(2026, 1, 1))

        self.assertEqual(provider_stats(provider)["login_count"], 1)
        self.assertEqual(provider_stats(provider)["history_query_count"], 1)
        self.assertEqual(provider_stats(provider)["logout_count"], 1)

    def test_session_reuses_login_for_multiple_symbols(self):
        with fake_baostock(FakeBaostock()):
            provider = BaostockProvider(cache=None, use_cache=False)
            with provider.session():
                for symbol in ("000001.SZ", "000002.SZ", "600000.SH"):
                    provider.get_bars(symbol, datetime(2026, 1, 1), datetime(2026, 1, 1))

        self.assertEqual(provider_stats(provider)["login_count"], 1)
        self.assertEqual(provider_stats(provider)["history_query_count"], 3)
        self.assertEqual(provider_stats(provider)["logout_count"], 1)

    def test_nested_session_does_not_double_login_or_early_logout(self):
        with fake_baostock(FakeBaostock()):
            provider = BaostockProvider(cache=None, use_cache=False)
            with provider.session():
                provider.get_bars("000001.SZ", datetime(2026, 1, 1), datetime(2026, 1, 1))
                with provider.session():
                    provider.get_bars("000002.SZ", datetime(2026, 1, 1), datetime(2026, 1, 1))
                    self.assertEqual(provider_stats(provider)["logout_count"], 0)

        self.assertEqual(provider_stats(provider)["login_count"], 1)
        self.assertEqual(provider_stats(provider)["history_query_count"], 2)
        self.assertEqual(provider_stats(provider)["logout_count"], 1)

    def test_session_with_all_cache_hits_never_logs_in(self):
        provider = BaostockProvider(cache=AlwaysHitCache(), use_cache=True)

        with provider.session():
            provider.get_bars("000001.SZ", datetime(2026, 1, 1), datetime(2026, 1, 1))
            provider.get_bars("000002.SZ", datetime(2026, 1, 1), datetime(2026, 1, 1))

        self.assertEqual(provider_stats(provider)["login_count"], 0)
        self.assertEqual(provider_stats(provider)["history_query_count"], 0)
        self.assertEqual(provider_stats(provider)["logout_count"], 0)
        self.assertEqual(provider_stats(provider)["cache_hit_count"], 2)

    def test_exception_restores_session_state_and_reraises(self):
        with fake_baostock(FakeBaostock(fail_history_call=2)):
            provider = BaostockProvider(cache=None, use_cache=False)
            with self.assertRaisesRegex(ValueError, "history boom"):
                with provider.session():
                    provider.get_bars("000001.SZ", datetime(2026, 1, 1), datetime(2026, 1, 1))
                    provider.get_bars("000002.SZ", datetime(2026, 1, 1), datetime(2026, 1, 1))

        self.assertEqual(provider_stats(provider)["login_count"], 1)
        self.assertEqual(provider_stats(provider)["history_query_count"], 2)
        self.assertEqual(provider_stats(provider)["logout_count"], 1)
        self.assertEqual(provider._session_depth, 0)
        self.assertFalse(provider._logged_in)

    def test_mixed_baostock_interfaces_share_one_session(self):
        with fake_baostock(FakeBaostock()):
            provider = BaostockProvider(cache=None, use_cache=False)
            with provider.session():
                provider.get_bars("000001.SZ", datetime(2026, 1, 1), datetime(2026, 1, 1))
                provider.get_trade_calendar(datetime(2026, 1, 1), datetime(2026, 1, 2))
                provider.get_stock_status("000001.SZ", datetime(2026, 1, 2))

        self.assertEqual(provider_stats(provider)["login_count"], 1)
        self.assertEqual(provider_stats(provider)["logout_count"], 1)
        self.assertEqual(provider_stats(provider)["history_query_count"], 1)
        self.assertEqual(provider_stats(provider)["trade_calendar_query_count"], 1)
        self.assertEqual(provider_stats(provider)["stock_basic_query_count"], 1)

    def test_feedback_groups_market_requests_by_symbol(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
            provider = CountingProvider()
            signals = []
            for symbol in ("000001.SZ", "000002.SZ"):
                for offset in range(3):
                    as_of = datetime(2026, 1, 10 + offset, 15, 0)
                    signals.append({
                        "signal_id": f"{symbol}-{offset}",
                        "symbol": symbol,
                        "stock_name": symbol,
                        "as_of_time": as_of.isoformat(),
                    })
            engine = FeedbackEngine(str(Path(directory) / "signals.sqlite"), provider)
            engine.store = FakeFeedbackStore(signals)
            summary = engine.update_all_pending()
            upsert_count = len(engine.store.upserts)
            gc.collect()

        self.assertEqual(provider.history_query_count, 2)
        self.assertEqual(summary["signals_found"], 6)
        self.assertEqual(summary["updated"], 6)
        self.assertEqual(upsert_count, 6)

    def test_tracking_groups_market_requests_by_symbol(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
            provider = CountingProvider()
            database_path = Path(directory) / "tracking.sqlite"
            SQLiteSignalStore(str(database_path))
            symbols = [
                "000001.SZ", "000001.SZ", "000001.SZ",
                "000002.SZ", "000002.SZ", "600000.SH",
                "600000.SH", "600000.SH",
            ]
            connection = sqlite3.connect(database_path)
            try:
                with connection:
                    for index, symbol in enumerate(symbols):
                        insert_tracking_row(connection, symbol, datetime(2020, 1, 10 + index, 15, 0))
            finally:
                connection.close()
            summary = RecommendationTracker(
                str(database_path), provider, "qfq"
            ).update_future_performance()
            gc.collect()

        self.assertEqual(provider.history_query_count, 3)
        self.assertEqual(summary["found"], 8)
        self.assertEqual(summary["updated"], 8)

    def test_symbol_name_resolver_resolve_many_reuses_session(self):
        with tempfile.TemporaryDirectory() as directory:
            with fake_baostock(FakeBaostock()):
                provider = BaostockProvider(cache=None, use_cache=False)
                resolver = SymbolNameResolver(
                    str(Path(directory) / "missing.sqlite"),
                    str(Path(directory) / "ai_candidates.json"),
                    baostock_provider=provider,
                )
                result = resolver.resolve_many(["000001.SZ", "000002.SZ", "600000.SH"])

        self.assertEqual(set(result), {"000001.SZ", "000002.SZ", "600000.SH"})
        self.assertEqual(provider_stats(provider)["login_count"], 1)
        self.assertEqual(provider_stats(provider)["stock_basic_query_count"], 3)
        self.assertEqual(provider_stats(provider)["logout_count"], 1)

    def test_a_share_coarse_scan_uses_one_baostock_session(self):
        with tempfile.TemporaryDirectory() as directory:
            with fake_baostock(FakeBaostock()):
                provider = BaostockProvider(cache=None, use_cache=False)
                reference_path = Path(directory) / "a_share_symbols.csv"
                pd.DataFrame(
                    [
                        {
                            "symbol": "000001.SZ",
                            "code": "000001",
                            "exchange": "SZ",
                            "stock_name": "A",
                            "asset_type": "stock",
                        },
                        {
                            "symbol": "000002.SZ",
                            "code": "000002",
                            "exchange": "SZ",
                            "stock_name": "B",
                            "asset_type": "stock",
                        },
                        {
                            "symbol": "600000.SH",
                            "code": "600000",
                            "exchange": "SH",
                            "stock_name": "C",
                            "asset_type": "stock",
                        },
                    ]
                ).to_csv(reference_path, index=False, encoding="utf-8-sig")
                scanner = AShareCoarseScanner(
                    SimpleNamespace(
                        market_provider="baostock",
                        market_frequency="1d",
                        market_adjust_type="qfq",
                    ),
                    market_data_provider=provider,
                    reference_path=str(reference_path),
                )
                scanner.run(max_symbols=3, limit=3)

        self.assertEqual(provider_stats(provider)["login_count"], 1)
        self.assertEqual(provider_stats(provider)["history_query_count"], 3)
        self.assertEqual(provider_stats(provider)["logout_count"], 1)


if __name__ == "__main__":
    unittest.main()
