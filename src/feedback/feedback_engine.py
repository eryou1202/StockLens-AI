from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from src.data.market_data_provider import MarketDataProvider
from src.feedback.feedback_schema import FutureLabel
from src.feedback.label_builder import FutureLabelBuilder
from src.storage.sqlite_store import SQLiteSignalStore


class FeedbackEngine:
    def __init__(
        self,
        database_path: str,
        market_data_provider: MarketDataProvider,
        lookahead_days: int = 30,
        adjust_type: str = "qfq",
    ) -> None:
        self.store = SQLiteSignalStore(database_path)
        self.market_data_provider = market_data_provider
        self.lookahead_days = lookahead_days
        self.adjust_type = adjust_type
        self.label_builder = FutureLabelBuilder()

    def update_all_pending(self, limit: int | None = None) -> dict[str, int]:
        signals = self.store.list_signals_for_feedback(limit=limit)
        summary = {
            "signals_found": len(signals),
            "updated": 0,
            "pending": 0,
            "partial": 0,
            "complete": 0,
            "failed": 0,
        }
        grouped: dict[str, list[tuple[dict[str, Any], datetime]]] = {}
        for signal in signals:
            symbol = str(signal.get("symbol") or "")
            stock_name = signal.get("stock_name")
            signal_id = str(signal.get("signal_id") or "")
            as_of_time_key = str(signal.get("as_of_time") or "")
            try:
                as_of_time = self._parse_datetime(signal.get("as_of_time"))
            except (TypeError, ValueError) as exc:
                label = FutureLabel(
                    symbol=symbol or "unknown",
                    stock_name=stock_name,
                    as_of_time=datetime.now(),
                    feedback_status="failed",
                    error_message=f"invalid_as_of_time: {exc}",
                )
                self._upsert_label(signal_id, label, 0, as_of_time_key or None)
                summary["updated"] += 1
                summary[label.feedback_status] += 1
                continue
            grouped.setdefault(symbol, []).append((signal, as_of_time))

        with self.market_data_provider.session():
            for symbol, items in grouped.items():
                start_time = min(as_of for _signal, as_of in items) - timedelta(days=10)
                end_time = max(as_of for _signal, as_of in items) + timedelta(days=self.lookahead_days)
                try:
                    market_data = self.market_data_provider.get_bars(
                        symbol=symbol,
                        start_time=start_time,
                        end_time=end_time,
                        frequency="1d",
                        adjust_type=self.adjust_type,
                    )
                except Exception as exc:
                    for signal, as_of_time in items:
                        label = FutureLabel(
                            symbol=symbol,
                            stock_name=signal.get("stock_name"),
                            as_of_time=as_of_time,
                            feedback_status="failed",
                            error_message=f"{type(exc).__name__}: {exc}",
                        )
                        self._upsert_label(
                            str(signal.get("signal_id") or ""),
                            label,
                            0,
                            str(signal.get("as_of_time") or "") or None,
                        )
                        summary["updated"] += 1
                        summary[label.feedback_status] += 1
                    continue

                for signal, as_of_time in items:
                    record_start = as_of_time - timedelta(days=10)
                    record_end = as_of_time + timedelta(days=self.lookahead_days)
                    record_market_data = self._slice_bundle(market_data, record_start, record_end)
                    label = self.label_builder.build(
                        symbol,
                        signal.get("stock_name"),
                        as_of_time,
                        record_market_data,
                    )
                    self._upsert_label(
                        str(signal.get("signal_id") or ""),
                        label,
                        len(record_market_data.bars),
                        str(signal.get("as_of_time") or "") or None,
                    )
                    summary["updated"] += 1
                    summary[label.feedback_status] += 1
        return summary

    def update_one_signal(self, signal: dict[str, Any]) -> FutureLabel:
        symbol = str(signal.get("symbol") or "")
        stock_name = signal.get("stock_name")
        signal_id = str(signal.get("signal_id") or "")
        as_of_time_key = str(signal.get("as_of_time") or "")
        try:
            as_of_time = self._parse_datetime(signal.get("as_of_time"))
        except (TypeError, ValueError) as exc:
            label = FutureLabel(
                symbol=symbol or "unknown",
                stock_name=stock_name,
                as_of_time=datetime.now(),
                feedback_status="failed",
                error_message=f"invalid_as_of_time: {exc}",
            )
            self._upsert_label(signal_id, label, 0, as_of_time_key or None)
            return label

        try:
            with self.market_data_provider.session():
                market_data = self.market_data_provider.get_bars(
                    symbol=symbol,
                    start_time=as_of_time - timedelta(days=10),
                    end_time=as_of_time + timedelta(days=self.lookahead_days),
                    frequency="1d",
                    adjust_type=self.adjust_type,
                )
            label = self.label_builder.build(symbol, stock_name, as_of_time, market_data)
            bar_count = len(market_data.bars)
        except Exception as exc:
            label = FutureLabel(
                symbol=symbol,
                stock_name=stock_name,
                as_of_time=as_of_time,
                feedback_status="failed",
                error_message=f"{type(exc).__name__}: {exc}",
            )
            bar_count = 0

        self._upsert_label(signal_id, label, bar_count, as_of_time_key or None)
        return label

    def _upsert_label(
        self,
        signal_id: str,
        label: FutureLabel,
        bar_count: int,
        as_of_time_key: str | None,
    ) -> None:
        self.store.upsert_feedback(
            signal_id,
            label,
            self._engine_metadata(bar_count),
            as_of_time_key=as_of_time_key,
        )

    @staticmethod
    def _slice_bundle(market_data, start_time: datetime, end_time: datetime):
        start = start_time.replace(tzinfo=None)
        end = end_time.replace(tzinfo=None)
        bars = [
            bar for bar in market_data.bars
            if start <= bar.trade_time.replace(tzinfo=None) <= end
        ]
        return market_data.model_copy(update={
            "start_time": start_time,
            "end_time": end_time,
            "bars": bars,
        })

    def _engine_metadata(self, bar_count: int) -> dict[str, Any]:
        return {
            "source": "FeedbackEngine v1.0",
            "lookahead_days": self.lookahead_days,
            "provider": self.market_data_provider.provider_name,
            "bar_count": bar_count,
        }

    @staticmethod
    def _parse_datetime(value: Any) -> datetime:
        if isinstance(value, datetime):
            return value
        if value is None:
            raise ValueError("missing as_of_time")
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
