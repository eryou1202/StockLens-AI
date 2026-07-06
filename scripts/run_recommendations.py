from __future__ import annotations

import argparse
from datetime import datetime

from rich.console import Console
from rich.table import Table

from src.ai.file_ai_info_engine import FileAIInfoEngine
from src.config.settings import load_settings
from src.data.provider_factory import create_market_data_provider
from src.decision.decision_engine import DecisionEngine
from src.pipeline import ResearchPipeline
from src.quant.quant_engine import RuleBasedQuantEngine
from src.recommendation.recommendation_engine import RecommendationEngine
from src.recommendation.recommendation_explainer import ACTION_LABELS
from src.recommendation.recommendation_schema import Recommendation
from src.reports.report_builder import ReportBuilder
from src.storage.sqlite_store import SQLiteSignalStore
from src.tracking.recommendation_tracker import RecommendationTracker


def run_recommendation_analysis(
    as_of_time: datetime | None = None,
) -> list[Recommendation]:
    settings = load_settings()
    market_provider = create_market_data_provider(
        name=settings.market_provider,
        cache_dir=settings.cache_dir,
        use_cache=True,
    )
    pipeline = ResearchPipeline(
        ai_engine=FileAIInfoEngine("data/ai_candidates.json"),
        market_data_provider=market_provider,
        quant_engine=RuleBasedQuantEngine(),
        decision_engine=DecisionEngine(),
        store=SQLiteSignalStore(settings.database_path),
        report_builder=ReportBuilder(),
        lookback_days=settings.default_lookback_days,
        frequency=settings.market_frequency,
        adjust_type=settings.market_adjust_type,
    )
    decisions = pipeline.run_once(as_of_time or datetime.now())
    return RecommendationEngine().build_many(decisions)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run StockLens recommendations.")
    parser.add_argument("--save-tracking", action="store_true", help="Save this recommendation snapshot.")
    args = parser.parse_args()
    try:
        recommendations = run_recommendation_analysis()
    except Exception as exc:
        print(f"推荐分析失败：{type(exc).__name__}: {exc}")
        return

    table = Table(title="StockLens 推荐候选（仅供研究，不构成投资建议）")
    if args.save_tracking:
        count = RecommendationTracker(load_settings().database_path).save_recommendations(recommendations)
        print(f"已保存追踪快照：{count} 条（重复快照自动跳过）。")

    for column in ("股票", "source_type", "action", "level", "confidence", "AI观点", "量化观点", "final_score", "主要理由"):
        table.add_column(column)
    for item in recommendations:
        table.add_row(
            f"{item.symbol} {item.stock_name or '未知名称'}",
            item.source_type,
            ACTION_LABELS[item.action],
            item.action_level.value,
            f"{item.confidence:.2f}",
            item.ai_view or "-",
            item.quant_decision or "-",
            "-" if item.final_score is None else f"{item.final_score:.1f}",
            item.reason[0] if item.reason else "-",
        )
    Console().print(table)


if __name__ == "__main__":
    main()
