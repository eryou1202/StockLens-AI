from __future__ import annotations

from datetime import datetime, time, timedelta
from typing import Callable

from src.audit.audit_schema import AuditRequest, AuditSample
from src.config.settings import AppSettings
from src.data.market_data_provider import MarketDataProvider
from src.data.provider_factory import create_market_data_provider
from src.decision.decision_engine import DecisionEngine
from src.feedback.label_builder import FutureLabelBuilder
from src.models.schemas import AICandidate, MarketDataBundle
from src.quant.quant_engine import RuleBasedQuantEngine
from src.recommendation.recommendation_engine import RecommendationEngine


ProgressCallback = Callable[[str, datetime, int, int], None]


class AlgorithmAuditRunner:
    """Runs leakage-safe quant audits without touching any production store."""

    def __init__(
        self,
        settings: AppSettings,
        market_data_provider: MarketDataProvider | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> None:
        self.settings = settings
        self.market_data_provider = market_data_provider or create_market_data_provider(
            settings.market_provider,
            cache_dir="data/audit/cache",
            use_cache=True,
        )
        self.progress_callback = progress_callback
        self.quant_engine = RuleBasedQuantEngine()
        self.decision_engine = DecisionEngine()
        self.recommendation_engine = RecommendationEngine()
        self.label_builder = FutureLabelBuilder()

    def run(self, request: AuditRequest, audit_id: int = 0) -> list[AuditSample]:
        symbols = request.symbols[: request.max_symbols] if request.max_symbols else request.symbols
        dates = self._audit_dates(request)
        total = len(symbols) * len(dates)
        samples: list[AuditSample] = []

        for symbol in symbols:
            try:
                full_bundle = self.market_data_provider.get_bars(
                    symbol=symbol,
                    start_time=request.start_date - timedelta(days=request.lookback_days),
                    end_time=request.end_date + timedelta(days=45),
                    frequency=self.settings.market_frequency,
                    adjust_type=self.settings.market_adjust_type,
                )
            except Exception as exc:
                for as_of_time in dates:
                    samples.append(self._error_sample(audit_id, symbol, as_of_time, exc))
                    self._progress(symbol, as_of_time, len(samples), total)
                continue

            for as_of_time in dates:
                try:
                    samples.append(
                        self._build_sample(audit_id, symbol, as_of_time, request, full_bundle)
                    )
                except Exception as exc:
                    samples.append(self._error_sample(audit_id, symbol, as_of_time, exc))
                self._progress(symbol, as_of_time, len(samples), total)
        return samples

    def _build_sample(
        self,
        audit_id: int,
        symbol: str,
        as_of_time: datetime,
        request: AuditRequest,
        full_bundle: MarketDataBundle,
    ) -> AuditSample:
        lookback_start = as_of_time - timedelta(days=request.lookback_days)
        past_bars = [
            bar for bar in full_bundle.bars
            if lookback_start <= self._align_datetime(bar.trade_time, as_of_time) <= as_of_time
        ]
        past_bundle = full_bundle.model_copy(update={
            "start_time": lookback_start,
            "end_time": as_of_time,
            "bars": past_bars,
            "metadata": {
                **full_bundle.metadata,
                "audit_past_only": True,
                "as_of_time": as_of_time.isoformat(),
            },
        })
        candidate = self._neutral_candidate(symbol, as_of_time)
        quant_result = self.quant_engine.analyze(candidate, past_bundle, as_of_time)
        decision = self.decision_engine.merge(candidate, quant_result)
        recommendation = self.recommendation_engine.build_from_decision(decision)
        context = self.recommendation_engine.extract_quant_context(decision)

        label = self.label_builder.build(symbol, None, as_of_time, full_bundle)
        return AuditSample(
            audit_id=audit_id,
            symbol=symbol,
            stock_name=None,
            as_of_time=as_of_time,
            mode=request.mode,
            action=recommendation.action.value,
            action_level=recommendation.action_level.value,
            final_score=decision.final_score,
            quant_score=context.get("quant_score"),
            quant_decision=context.get("quant_decision"),
            ai_view=decision.ai_view,
            current_price=context.get("latest_close"),
            future_return_1d=label.future_return_1d,
            future_return_3d=label.future_return_3d,
            future_return_5d=label.future_return_5d,
            future_return_10d=label.future_return_10d,
            future_max_drawdown_5d=label.future_max_drawdown_5d,
            future_max_drawdown_10d=label.future_max_drawdown_10d,
            trend_score=context.get("trend_score"),
            momentum_score=context.get("momentum_score"),
            volume_score=context.get("volume_score"),
            risk_score=context.get("risk_score"),
            overheat_score=context.get("overheat_score"),
            macd_score=context.get("macd_score"),
            return_5d=context.get("return_5d"),
            return_20d=context.get("return_20d"),
            close_ma20_gap=context.get("close_ma20_gap"),
            ma5_ma20_gap=context.get("ma5_ma20_gap"),
            ma20_ma60_gap=context.get("ma20_ma60_gap"),
            rsi_14=context.get("rsi_14"),
            macd_hist=context.get("macd_hist"),
            volume_ratio_5d=context.get("volume_ratio_5d"),
            max_drawdown_20d=context.get("max_drawdown_20d"),
            volatility_20d=context.get("volatility_20d"),
            sample_note=(
                "quant_only_audit；推荐仅使用审查日及之前行情；"
                f"未来标签状态={label.feedback_status}；"
                f"入场对齐规则={label.metadata.get('entry_rule', 'unknown')}"
            ),
            error_message=label.error_message if label.feedback_status == "failed" else None,
        )

    @staticmethod
    def _neutral_candidate(symbol: str, as_of_time: datetime) -> AICandidate:
        return AICandidate(
            stock_code=symbol,
            stock_name=None,
            as_of_time=as_of_time,
            event_time=as_of_time,
            event_type="quant_audit",
            event_summary="算法审查使用中性信息输入，只评估量化框架。",
            sentiment_score=0.0,
            event_strength=0.0,
            source_confidence=0.0,
            ai_confidence=0.2,
            risk_flags=[],
            source_urls=[],
            metadata={"source_type": "quant_audit", "audit": True},
        )

    @staticmethod
    def _audit_dates(request: AuditRequest) -> list[datetime]:
        result: list[datetime] = []
        current = request.start_date
        while current <= request.end_date:
            result.append(datetime.combine(current.date(), time(15, 0), tzinfo=current.tzinfo))
            current += timedelta(days=request.step_days)
        return result

    @staticmethod
    def _align_datetime(value: datetime, reference: datetime) -> datetime:
        if reference.tzinfo is None:
            return value.replace(tzinfo=None)
        if value.tzinfo is None:
            return value.replace(tzinfo=reference.tzinfo)
        return value.astimezone(reference.tzinfo)

    @staticmethod
    def _error_sample(audit_id: int, symbol: str, as_of_time: datetime, exc: Exception) -> AuditSample:
        return AuditSample(
            audit_id=audit_id,
            symbol=symbol,
            as_of_time=as_of_time,
            sample_note="该样本失败，但没有中断其他股票或日期。",
            error_message=f"{type(exc).__name__}: {exc}",
        )

    def _progress(self, symbol: str, as_of_time: datetime, completed: int, total: int) -> None:
        if self.progress_callback:
            self.progress_callback(symbol, as_of_time, completed, total)
        else:
            print(
                f"审查进度 {completed}/{total} | {symbol} | {as_of_time.date().isoformat()}",
                flush=True,
            )
