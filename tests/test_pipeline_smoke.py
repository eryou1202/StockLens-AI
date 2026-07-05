from datetime import datetime

from src.ai.ai_info_engine import MockAIInfoEngine
from src.data.market_data_provider import MockMarketDataProvider
from src.decision.decision_engine import DecisionEngine
from src.pipeline import ResearchPipeline
from src.quant.quant_engine import RuleBasedQuantEngine
from src.reports.report_builder import ReportBuilder
from src.storage.sqlite_store import SQLiteSignalStore


def test_pipeline_smoke(tmp_path):
    pipeline = ResearchPipeline(
        ai_engine=MockAIInfoEngine(),
        market_data_provider=MockMarketDataProvider(),
        quant_engine=RuleBasedQuantEngine(),
        decision_engine=DecisionEngine(),
        store=SQLiteSignalStore(str(tmp_path / "signals.sqlite")),
        report_builder=ReportBuilder(),
        lookback_days=120,
    )

    decisions = pipeline.run_once(datetime.now())
    assert len(decisions) > 0
    assert decisions[0].symbol
