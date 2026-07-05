from __future__ import annotations

from datetime import datetime

from src.ai.file_ai_info_engine import FileAIInfoEngine
from src.config.settings import load_settings
from src.data.provider_factory import create_market_data_provider
from src.decision.decision_engine import DecisionEngine
from src.pipeline import ResearchPipeline
from src.quant.quant_engine import RuleBasedQuantEngine
from src.reports.report_builder import ReportBuilder
from src.storage.sqlite_store import SQLiteSignalStore


def main() -> None:
    settings = load_settings()

    ai_engine = FileAIInfoEngine("data/ai_candidates.json")

    market_provider = create_market_data_provider(
        name=settings.market_provider,
        cache_dir=settings.cache_dir,
        use_cache=True,
    )

    quant_engine = RuleBasedQuantEngine()
    decision_engine = DecisionEngine()
    store = SQLiteSignalStore(settings.database_path)
    report_builder = ReportBuilder()

    pipeline = ResearchPipeline(
        ai_engine=ai_engine,
        market_data_provider=market_provider,
        quant_engine=quant_engine,
        decision_engine=decision_engine,
        store=store,
        report_builder=report_builder,
        lookback_days=settings.default_lookback_days,
        frequency=settings.market_frequency,
        adjust_type=settings.market_adjust_type,
    )

    decisions = pipeline.run_once(datetime.now())
    report_builder.print_report(decisions)

    print(f"\nAI 候选池文件：data/ai_candidates.json")
    print(f"当前行情数据源：{market_provider.provider_name}")
    print(f"已保存到数据库：{settings.database_path}")


if __name__ == "__main__":
    main()