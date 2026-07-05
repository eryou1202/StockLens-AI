from __future__ import annotations

import argparse
from datetime import datetime

from src.ai.file_ai_info_engine import FileAIInfoEngine
from src.config.settings import load_settings
from src.data.provider_factory import create_market_data_provider
from src.decision.decision_engine import DecisionEngine
from src.pipeline import ResearchPipeline
from src.quant.quant_engine import RuleBasedQuantEngine
from src.reports.report_builder import ReportBuilder
from src.storage.sqlite_store import SQLiteSignalStore


def parse_as_of(value: str) -> datetime:
    text = value.strip()
    try:
        if len(text) == 10:
            # 仅传日期时按日线收盘时点运行，允许使用当日已完成的 K 线。
            return datetime.fromisoformat(text).replace(hour=15, minute=0, second=0, microsecond=0)
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--as-of 格式应为 YYYY-MM-DD 或 ISO datetime"
        ) from exc


def main() -> None:
    parser = argparse.ArgumentParser(description="Run StockLens candidates at a historical time.")
    parser.add_argument(
        "--as-of",
        required=True,
        type=parse_as_of,
        help="历史分析时点，例如 2026-06-10 或 2026-06-10T14:30:00",
    )
    args = parser.parse_args()
    as_of_time: datetime = args.as_of

    settings = load_settings()
    ai_engine = FileAIInfoEngine("data/ai_candidates.json")
    market_provider = create_market_data_provider(
        name=settings.market_provider,
        cache_dir=settings.cache_dir,
        use_cache=True,
    )
    report_builder = ReportBuilder()
    pipeline = ResearchPipeline(
        ai_engine=ai_engine,
        market_data_provider=market_provider,
        quant_engine=RuleBasedQuantEngine(),
        decision_engine=DecisionEngine(),
        store=SQLiteSignalStore(settings.database_path),
        report_builder=report_builder,
        lookback_days=settings.default_lookback_days,
        frequency=settings.market_frequency,
        adjust_type=settings.market_adjust_type,
    )

    decisions = pipeline.run_once(as_of_time)
    report_builder.print_report(decisions)

    print(f"\nas_of_time: {as_of_time.isoformat()}")
    print(f"候选股票数量：{len(decisions)}")
    print("AI 候选池文件：data/ai_candidates.json")
    print(f"当前行情数据源：{market_provider.provider_name}")
    print(f"数据库路径：{settings.database_path}")


if __name__ == "__main__":
    main()
