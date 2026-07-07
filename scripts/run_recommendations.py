from __future__ import annotations

import argparse
from datetime import datetime
from typing import Any

from rich.console import Console
from rich.table import Table

from src.ai.file_ai_info_engine import FileAIInfoEngine
from src.config.settings import load_settings
from src.data.provider_factory import create_market_data_provider
from src.data.realtime_quote_provider import RealtimeQuoteProvider
from src.decision.decision_engine import DecisionEngine
from src.models.schemas import FinalDecision, MarketBar
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
    force_realtime_refresh: bool = True,
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
    engine = RecommendationEngine()
    recommendations = engine.build_many(decisions)

    # Fetch the whole A-share spot table once, then overlay each recommendation.
    # Daily indicators remain unchanged; the quote is only used for intraday confirmation.
    quote_provider = RealtimeQuoteProvider(cache_seconds=30)
    quotes = quote_provider.get_all_quotes(force_refresh=force_realtime_refresh)
    decision_map = {item.symbol: item for item in decisions}
    return [
        engine.apply_intraday_overlay(
            item,
            quote_provider.get_current_price(
                item.symbol,
                latest_market_bar=_daily_fallback_bar(decision_map.get(item.symbol)),
                quotes=quotes,
            ),
        )
        for item in recommendations
    ]


def _daily_fallback_bar(decision: FinalDecision | None) -> MarketBar | None:
    if decision is None:
        return None
    context = RecommendationEngine.extract_quant_context(decision)
    close = _number(context.get("latest_close"))
    if close is None or close <= 0:
        return None
    vector = context.get("feature_vector") if isinstance(context.get("feature_vector"), dict) else {}
    metadata = vector.get("metadata") if isinstance(vector.get("metadata"), dict) else {}
    trade_time = _datetime(metadata.get("latest_trade_time")) or decision.as_of_time
    return MarketBar(
        symbol=decision.symbol,
        trade_time=trade_time,
        close=close,
        provider="quant_daily_fallback",
    )


def _number(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value)) if value else None
    except (TypeError, ValueError):
        return None


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

    for column in (
        "股票", "source_type", "数据类型", "实时价", "实时涨跌幅", "盘中确认",
        "原始动作", "当前动作", "level", "confidence", "AI观点", "量化观点",
        "final_score", "数据时间", "主要理由",
    ):
        table.add_column(column)
    for item in recommendations:
        meta = item.metadata
        original_action = meta.get("original_action")
        try:
            original_label = ACTION_LABELS[type(item.action)(original_action)]
        except (TypeError, ValueError, KeyError):
            original_label = str(original_action or "-")
        table.add_row(
            f"{item.symbol} {item.stock_name or '未知名称'}",
            item.source_type,
            "实时行情" if meta.get("is_realtime") else "非实时 / 最新日线",
            "-" if meta.get("realtime_price") is None else f"{meta['realtime_price']:.2f}",
            "-" if meta.get("realtime_pct_change") is None else f"{meta['realtime_pct_change']:.2%}",
            "是" if meta.get("intraday_confirmed") else "否",
            original_label,
            ACTION_LABELS[item.action],
            item.action_level.value,
            f"{item.confidence:.2f}",
            item.ai_view or "-",
            item.quant_decision or "-",
            "-" if item.final_score is None else f"{item.final_score:.1f}",
            str(meta.get("price_time") or "-"),
            item.reason[0] if item.reason else "-",
        )
    Console().print(table)


if __name__ == "__main__":
    main()
