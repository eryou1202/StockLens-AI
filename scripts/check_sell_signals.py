from __future__ import annotations

from datetime import datetime, timedelta

from rich.console import Console
from rich.table import Table

from src.ai.file_ai_info_engine import FileAIInfoEngine
from src.config.settings import load_settings
from src.data.provider_factory import create_market_data_provider
from src.decision.decision_engine import DecisionEngine
from src.models.schemas import AICandidate
from src.portfolio.position_manager import PositionManager
from src.quant.quant_engine import RuleBasedQuantEngine
from src.recommendation.recommendation_explainer import ACTION_LABELS
from src.recommendation.recommendation_schema import Recommendation
from src.recommendation.sell_signal_engine import SellSignalEngine


def build_open_position_sell_signals(
    as_of_time: datetime | None = None,
) -> list[Recommendation]:
    settings = load_settings()
    positions = PositionManager(settings.database_path).list_positions("open")
    if not positions:
        return []
    now = as_of_time or datetime.now()
    candidates = FileAIInfoEngine("data/ai_candidates.json").generate_candidates(now)
    candidate_map = {candidate.stock_code: candidate for candidate in candidates}
    provider = create_market_data_provider(
        settings.market_provider,
        cache_dir=settings.cache_dir,
        use_cache=True,
    )
    quant_engine = RuleBasedQuantEngine()
    decision_engine = DecisionEngine()
    sell_engine = SellSignalEngine()
    results: list[Recommendation] = []

    for position in positions:
        candidate = candidate_map.get(position.symbol) or AICandidate(
            stock_code=position.symbol,
            stock_name=position.stock_name,
            as_of_time=now,
            event_time=now,
            event_type="position_monitor",
            event_summary="持仓监控使用中性 AI 信息，仅检查量价和风险条件。",
            sentiment_score=0.0,
            event_strength=0.0,
            source_confidence=0.0,
            ai_confidence=0.2,
            risk_flags=["该持仓不在当前 AI 候选池中。"],
        )
        market_data = provider.get_bars(
            symbol=position.symbol,
            start_time=now - timedelta(days=settings.default_lookback_days),
            end_time=now,
            frequency=settings.market_frequency,
            adjust_type=settings.market_adjust_type,
        )
        quant_result = quant_engine.analyze(candidate, market_data, now)
        decision = decision_engine.merge(candidate, quant_result)
        results.append(sell_engine.build_sell_signal(position, decision))
    return results


def main() -> None:
    try:
        signals = build_open_position_sell_signals()
    except Exception as exc:
        print(f"卖出提醒检查失败：{type(exc).__name__}: {exc}")
        return
    if not signals:
        print("当前没有 open 持仓。")
        return

    table = Table(title="StockLens 持仓卖出提醒（不构成自动交易指令）")
    for column in ("股票", "action", "level", "当前价", "成本价", "浮动收益", "量化观点", "主要理由"):
        table.add_column(column)
    for item in signals:
        metadata = item.metadata
        table.add_row(
            f"{item.symbol} {item.stock_name or ''}",
            ACTION_LABELS[item.action],
            item.action_level.value,
            "-" if metadata.get("current_price") is None else f"{metadata['current_price']:.4f}",
            "-" if metadata.get("entry_price") is None else f"{metadata['entry_price']:.4f}",
            "-" if metadata.get("unrealized_return") is None else f"{metadata['unrealized_return']:.2%}",
            item.quant_decision or "-",
            item.reason[0] if item.reason else "-",
        )
    Console().print(table)


if __name__ == "__main__":
    main()
