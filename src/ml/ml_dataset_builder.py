from __future__ import annotations

import hashlib
from datetime import datetime, time, timedelta
from typing import Callable

from src.config.settings import AppSettings
from src.data.market_data_provider import MarketDataProvider
from src.data.provider_factory import create_market_data_provider
from src.ml.ml_context_features import MLContextFeatureBuilder
from src.ml.ml_feature_builder import MLFeatureBuilder
from src.ml.ml_label_builder import MLLabelBuilder
from src.ml.ml_schema import MLDatasetRequest, MLResearchSample
from src.models.schemas import MarketDataBundle
from src.reference.symbol_reference import AShareSymbolReference


ProgressCallback = Callable[[str, datetime, int, int], None]


class MLDatasetBuilder:
    """Build isolated research samples with explicit past/future boundaries."""

    def __init__(
        self,
        settings: AppSettings,
        market_data_provider: MarketDataProvider | None = None,
        progress_callback: ProgressCallback | None = None,
        use_cache: bool = True,
        cache_dir: str = "data/ml/cache",
    ) -> None:
        self.settings = settings
        if market_data_provider is not None:
            self.provider = market_data_provider
            self.context_provider = market_data_provider
        else:
            self.provider = create_market_data_provider(
                settings.market_provider,
                cache_dir=cache_dir,
                use_cache=use_cache,
            )
            # Index context must cover the full requested range. The legacy CSV cache
            # returns any overlapping slice, so a short smoke cache could otherwise
            # hide earlier index history.
            self.context_provider = create_market_data_provider(
                settings.market_provider,
                cache_dir="data/ml/context_cache",
                use_cache=False,
            )
        self.feature_builder = MLFeatureBuilder()
        self.label_builder = MLLabelBuilder()
        self.progress_callback = progress_callback
        reference = AShareSymbolReference()
        self.name_map = reference.name_map()
        self.asset_types = reference.asset_type_map()

    def build(self, request: MLDatasetRequest) -> list[MLResearchSample]:
        symbols = request.symbols[: request.max_symbols] if request.max_symbols else request.symbols
        symbols = [
            symbol for symbol in symbols
            if self.asset_types.get(symbol, "stock") == "stock"
        ]
        completed = 0
        samples: list[MLResearchSample] = []
        max_horizon = max(request.horizons) if request.horizons else 1
        prepared: list[tuple[str, MarketDataBundle | None, list[datetime], Exception | None]] = []

        for symbol in symbols:
            try:
                full_bundle = self.provider.get_bars(
                    symbol=symbol,
                    start_time=request.start_date - timedelta(days=request.lookback_days + 10),
                    end_time=request.end_date + timedelta(days=max_horizon * 3 + 15),
                    frequency=self.settings.market_frequency,
                    adjust_type=self.settings.market_adjust_type,
                )
                dates = self._sample_dates(request, full_bundle)
                if not dates:
                    raise ValueError("no trading bars in requested date range")
                prepared.append((symbol, full_bundle, dates, None))
            except Exception as exc:
                prepared.append((symbol, None, [self._failure_as_of(request)], exc))

        total = sum(len(dates) for _symbol, _bundle, dates, _error in prepared)
        context_builder: MLContextFeatureBuilder | None = None
        context_symbols: list[str] = []
        if request.include_context_features:
            context_bundles = {
                symbol: bundle
                for symbol, bundle, _dates, error in prepared
                if bundle is not None and error is None
            }
            context_symbols = list(context_bundles)
            context_builder = MLContextFeatureBuilder(
                provider=self.context_provider,
                frequency=self.settings.market_frequency,
                adjust_type=self.settings.market_adjust_type,
                lookback_days=request.lookback_days,
            )
            context_builder.prepare(
                symbol_bundles=context_bundles,
                start_date=request.start_date,
                end_date=request.end_date,
            )
        for symbol, full_bundle, dates, fetch_error in prepared:
            for as_of in dates:
                if fetch_error is not None or full_bundle is None:
                    samples.append(self._failed_sample(
                        symbol,
                        as_of,
                        request,
                        fetch_error or RuntimeError("market data unavailable"),
                    ))
                else:
                    try:
                        samples.append(self._build_sample(
                            symbol,
                            as_of,
                            request,
                            full_bundle,
                            context_builder,
                            context_symbols,
                        ))
                    except Exception as exc:
                        samples.append(self._failed_sample(symbol, as_of, request, exc))
                completed += 1
                self._progress(symbol, as_of, completed, total)
        return samples

    def _build_sample(
        self,
        symbol: str,
        as_of: datetime,
        request: MLDatasetRequest,
        full_bundle: MarketDataBundle,
        context_builder: MLContextFeatureBuilder | None = None,
        context_symbols: list[str] | None = None,
    ) -> MLResearchSample:
        lookback_start = as_of - timedelta(days=request.lookback_days)
        past_bars = [
            bar for bar in full_bundle.sorted_bars()
            if lookback_start <= self._align(bar.trade_time, as_of) <= as_of
        ]
        if not past_bars:
            raise ValueError("no past bars available")
        past_bundle = full_bundle.model_copy(update={
            "start_time": lookback_start,
            "end_time": as_of,
            "bars": past_bars,
            "metadata": {
                **full_bundle.metadata,
                "ml_research": True,
                "past_only": True,
                "as_of_time": as_of.isoformat(),
            },
        })
        feature_values = self.feature_builder.build(past_bundle)
        context_warnings: list[str] = []
        if context_builder is not None:
            try:
                feature_values.update(context_builder.build(
                    as_of_date=as_of,
                    symbol=symbol,
                    symbols=context_symbols or [],
                    stock_features=feature_values,
                ))
                context_warnings = context_builder.warnings_for(as_of)
            except Exception as exc:
                feature_values.update(context_builder.empty_features())
                context_warnings = [f"context_build_failed:{type(exc).__name__}:{exc}"]
        label_values = self.label_builder.build(full_bundle, as_of, request.horizons)
        return MLResearchSample(
            sample_id=self._sample_id(symbol, as_of),
            symbol=symbol,
            stock_name=self.name_map.get(symbol),
            as_of_date=as_of.date().isoformat(),
            sample_interval_days=request.sample_interval_days,
            lookback_days=request.lookback_days,
            **feature_values,
            **label_values,
            metadata={
                "past_bar_count": len(past_bars),
                "past_only": True,
                "include_context_features": request.include_context_features,
                "context_warnings": context_warnings,
            },
        )

    def _failed_sample(
        self,
        symbol: str,
        as_of: datetime,
        request: MLDatasetRequest,
        exc: Exception,
    ) -> MLResearchSample:
        return MLResearchSample(
            sample_id=self._sample_id(symbol, as_of),
            symbol=symbol,
            stock_name=self.name_map.get(symbol),
            as_of_date=as_of.date().isoformat(),
            source=self.settings.market_provider,
            sample_interval_days=request.sample_interval_days,
            lookback_days=request.lookback_days,
            label_status="failed",
            label_error=f"{type(exc).__name__}: {exc}",
            metadata={
                "past_only": True,
                "include_context_features": request.include_context_features,
            },
        )

    @classmethod
    def _sample_dates(
        cls,
        request: MLDatasetRequest,
        full_bundle: MarketDataBundle,
    ) -> list[datetime]:
        trading_dates: list[datetime] = []
        seen_dates = set()
        for bar in full_bundle.sorted_bars():
            aligned = cls._align(bar.trade_time, request.start_date)
            trade_date = aligned.date()
            if not request.start_date.date() <= trade_date <= request.end_date.date():
                continue
            if bar.trade_status is not None and str(bar.trade_status).strip() != "1":
                continue
            if trade_date in seen_dates:
                continue
            seen_dates.add(trade_date)
            trading_dates.append(
                datetime.combine(trade_date, time(15, 0), tzinfo=request.start_date.tzinfo)
            )
        return trading_dates[:: request.sample_interval_days]

    @staticmethod
    def _failure_as_of(request: MLDatasetRequest) -> datetime:
        return datetime.combine(
            request.start_date.date(),
            time(15, 0),
            tzinfo=request.start_date.tzinfo,
        )

    @staticmethod
    def _sample_id(symbol: str, as_of: datetime) -> str:
        raw = f"{symbol}|{as_of.isoformat()}".encode("utf-8")
        return hashlib.sha256(raw).hexdigest()[:20]

    @staticmethod
    def _align(value: datetime, reference: datetime) -> datetime:
        if reference.tzinfo is None:
            return value.replace(tzinfo=None)
        if value.tzinfo is None:
            return value.replace(tzinfo=reference.tzinfo)
        return value.astimezone(reference.tzinfo)

    def _progress(self, symbol: str, as_of: datetime, completed: int, total: int) -> None:
        if self.progress_callback:
            self.progress_callback(symbol, as_of, completed, total)
        else:
            print(f"ML dataset {completed}/{total} | {symbol} | {as_of.date().isoformat()}", flush=True)
