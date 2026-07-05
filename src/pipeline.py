from __future__ import annotations

from datetime import datetime, timedelta

from src.ai.ai_info_engine import AIInfoEngine
from src.data.market_data_provider import MarketDataProvider
from src.decision.decision_engine import DecisionEngine
from src.models.schemas import FinalDecision
from src.quant.quant_engine import QuantEngine
from src.reports.report_builder import ReportBuilder
from src.storage.sqlite_store import SQLiteSignalStore


class ResearchPipeline:
    """
    项目主流程。

    AI 推荐池
    → MarketDataProvider 读取行情
    → QuantEngine 分析曲线
    → DecisionEngine 融合判断
    → 保存和展示
    """

    def __init__(
        self,
        ai_engine: AIInfoEngine,
        market_data_provider: MarketDataProvider,
        quant_engine: QuantEngine,
        decision_engine: DecisionEngine,
        store: SQLiteSignalStore,
        report_builder: ReportBuilder,
        lookback_days: int = 120,
        frequency: str = "1d",
        adjust_type: str = "qfq",
    ):
        self.ai_engine = ai_engine
        self.market_data_provider = market_data_provider
        self.quant_engine = quant_engine
        self.decision_engine = decision_engine
        self.store = store
        self.report_builder = report_builder
        self.lookback_days = lookback_days
        self.frequency = frequency
        self.adjust_type = adjust_type

    def run_once(self, as_of_time: datetime) -> list[FinalDecision]:
        candidates = self.ai_engine.generate_candidates(as_of_time)
        final_decisions: list[FinalDecision] = []

        start_time = as_of_time - timedelta(days=self.lookback_days)

        for candidate in candidates:
            market_data = self.market_data_provider.get_bars(
                symbol=candidate.stock_code,
                start_time=start_time,
                end_time=as_of_time,
                frequency=self.frequency,
                adjust_type=self.adjust_type,
            )

            quant_result = self.quant_engine.analyze(
                candidate=candidate,
                market_data=market_data,
                as_of_time=as_of_time,
            )

            final_decision = self.decision_engine.merge(
                ai_signal=candidate,
                quant_signal=quant_result,
            )

            self.store.save_decision(final_decision)
            final_decisions.append(final_decision)

        return self.report_builder.build_watchlist(final_decisions)
