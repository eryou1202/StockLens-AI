from __future__ import annotations

import argparse
import math
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


def build_open_position_sell_signals(
    as_of_time: datetime | None = None,
    use_cache: bool = False,
    force_refresh: bool = True,
) -> list[Recommendation]:
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
    # force_refresh wins over use_cache. Providers without a dedicated refresh API
    # are refreshed by constructing them with cache reads/writes disabled.
    provider = create_market_data_provider(
        settings.market_provider,
        cache_dir=settings.cache_dir,
        use_cache=bool(use_cache and not force_refresh),
    )
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
        current_price, price_time = _latest_market_price(market_data)
        if not position.stock_name:
            inferred = _infer_stock_name(market_data)
            if inferred and position.id:
                manager.update_stock_name(position.id, inferred)
                position = position.model_copy(update={"stock_name": inferred})
        decision = decision_engine.merge(candidate, quant_engine.analyze(candidate, market_data, now))
        results.append(
            sell_engine.build_sell_signal(
                position,
                decision,
                current_price=current_price,
                price_time=price_time,
                price_source="latest_market_bar" if current_price is not None else "unavailable",
                require_fresh_price=True,
            )
        )
    return results


def _latest_market_price(bundle) -> tuple[float | None, str | None]:
    """Return the newest valid close from this fetch, never from old recommendations."""
    for bar in reversed(bundle.sorted_bars()):
        try:
            close = float(bar.close)
        except (TypeError, ValueError, OverflowError):
            continue
        if math.isfinite(close) and close > 0:
            return close, bar.trade_time.isoformat()
    return None, None


def _infer_stock_name(bundle) -> str | None:
    for bar in reversed(bundle.sorted_bars()):
        raw = bar.raw if isinstance(bar.raw, dict) else {}
        for key in ("stock_name", "name", "证券名称", "名称"):
            value = raw.get(key)
            if value:
                return str(value).strip()
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Check positions with freshly fetched market data.")
    cache_group = parser.add_mutually_exclusive_group()
    cache_group.add_argument("--use-cache", dest="use_cache", action="store_true")
    cache_group.add_argument("--no-cache", dest="use_cache", action="store_false")
    parser.set_defaults(use_cache=False)
    args = parser.parse_args()
    try:
        signals = build_open_position_sell_signals(
            use_cache=args.use_cache,
            force_refresh=not args.use_cache,
        )
    except Exception as exc:
        print(f"卖出提醒检查失败：{type(exc).__name__}: {exc}")
        return
    if not signals:
        print("当前没有 open 持仓或 watch_only 观察股。")
        return
    table = Table(title="StockLens 持仓/观察提醒（不构成自动交易指令）")
    for column in ("股票", "status", "action", "level", "当前价", "价格时间", "成本价", "浮动收益", "触发规则", "主要理由"):
        table.add_column(column)
    for item in signals:
        meta = item.metadata
        table.add_row(
            f"{item.symbol} {item.stock_name or '未知名称'}", str(meta.get("position_status", "-")),
            ACTION_LABELS[item.action], item.action_level.value,
            "未获取" if meta.get("current_price") is None else f"{meta['current_price']:.2f}",
            str(meta.get("price_time") or "未获取"),
            "-" if meta.get("is_watch_only") else f"{meta['entry_price']:.2f}",
            "-" if meta.get("unrealized_return") is None else f"{meta['unrealized_return']:.2%}",
            "；".join(meta.get("triggered_rule_labels", [])), "；".join(item.reason),
        )
    Console().print(table)


if __name__ == "__main__":
    main()
