from __future__ import annotations

from datetime import datetime, timedelta
from rich.console import Console
from rich.table import Table
from src.ai.file_ai_info_engine import FileAIInfoEngine
from src.config.settings import load_settings
from src.data.provider_factory import create_market_data_provider
from src.data.symbol_name_resolver import SymbolNameResolver
from src.decision.decision_engine import DecisionEngine
from src.models.schemas import AICandidate
from src.portfolio.position_manager import PositionManager
from src.quant.quant_engine import RuleBasedQuantEngine
from src.recommendation.recommendation_explainer import ACTION_LABELS
from src.recommendation.recommendation_schema import Recommendation
from src.recommendation.sell_signal_engine import SellSignalEngine


def build_open_position_sell_signals(as_of_time: datetime | None = None) -> list[Recommendation]:
    """Compatibility name; v1.1 monitors both open and watch_only records."""
    settings = load_settings()
    manager = PositionManager(settings.database_path)
    name_resolver = SymbolNameResolver(settings.database_path, "data/ai_candidates.json")
    positions = manager.list_positions("active")
    if not positions:
        return []
    now = as_of_time or datetime.now()
    candidates = FileAIInfoEngine("data/ai_candidates.json").generate_candidates(now)
    candidate_map = {item.stock_code: item for item in candidates}
    provider = create_market_data_provider(settings.market_provider, cache_dir=settings.cache_dir, use_cache=True)
    quant_engine, decision_engine, sell_engine = RuleBasedQuantEngine(), DecisionEngine(), SellSignalEngine()
    results: list[Recommendation] = []
    for position in positions:
        if not position.stock_name or position.stock_name.strip() in {"", "-", "未知名称"}:
            resolved_name = name_resolver.update_position_name_if_missing(position.symbol)
            if resolved_name:
                position = position.model_copy(update={"stock_name": resolved_name})
        candidate = candidate_map.get(position.symbol)
        if candidate and not position.stock_name and candidate.stock_name and position.id:
            manager.update_stock_name(position.id, candidate.stock_name)
            position = position.model_copy(update={"stock_name": candidate.stock_name})
        candidate = candidate or AICandidate(
            stock_code=position.symbol, stock_name=position.stock_name, as_of_time=now,
            event_time=now, event_type="position_monitor",
            event_summary="持仓/观察监控使用中性 AI 信息，仅检查量价和风险条件。",
            sentiment_score=0.0, event_strength=0.0, source_confidence=0.0, ai_confidence=0.2,
            risk_flags=["该标的不在当前 AI 候选池中。"],
            metadata={"source_type": position.metadata.get("source_type", "position")},
        )
        market_data = provider.get_bars(
            position.symbol, now - timedelta(days=settings.default_lookback_days), now,
            settings.market_frequency, settings.market_adjust_type,
        )
        if not position.stock_name:
            inferred = _infer_stock_name(market_data)
            if inferred and position.id:
                manager.update_stock_name(position.id, inferred)
                position = position.model_copy(update={"stock_name": inferred})
        decision = decision_engine.merge(candidate, quant_engine.analyze(candidate, market_data, now))
        results.append(sell_engine.build_sell_signal(position, decision))
    return results


def _infer_stock_name(bundle) -> str | None:
    for bar in reversed(bundle.sorted_bars()):
        raw = bar.raw if isinstance(bar.raw, dict) else {}
        for key in ("stock_name", "name", "证券名称", "名称"):
            value = raw.get(key)
            if value:
                return str(value).strip()
    return None


def main() -> None:
    try:
        signals = build_open_position_sell_signals()
    except Exception as exc:
        print(f"卖出提醒检查失败：{type(exc).__name__}: {exc}")
        return
    if not signals:
        print("当前没有 open 持仓或 watch_only 观察股。")
        return
    table = Table(title="StockLens 持仓/观察提醒（不构成自动交易指令）")
    for column in ("股票", "status", "action", "level", "当前价", "成本价", "浮动收益", "触发规则", "主要理由"):
        table.add_column(column)
    for item in signals:
        meta = item.metadata
        table.add_row(
            f"{item.symbol} {item.stock_name or '未知名称'}", str(meta.get("position_status", "-")),
            ACTION_LABELS[item.action], item.action_level.value,
            "-" if meta.get("current_price") is None else f"{meta['current_price']:.2f}",
            "-" if meta.get("is_watch_only") else f"{meta['entry_price']:.2f}",
            "-" if meta.get("unrealized_return") is None else f"{meta['unrealized_return']:.2%}",
            "；".join(meta.get("triggered_rule_labels", [])), "；".join(item.reason),
        )
    Console().print(table)


if __name__ == "__main__":
    main()
