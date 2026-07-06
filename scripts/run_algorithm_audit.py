from __future__ import annotations

import argparse
from datetime import datetime, time

from rich.console import Console
from rich.table import Table

from src.audit.algorithm_audit import AlgorithmAuditRunner
from src.audit.audit_metrics import AuditMetricsBuilder
from src.audit.audit_schema import AuditRequest
from src.audit.audit_store import AuditStore
from src.audit.universe_loader import SMALL_DEMO_UNIVERSE, load_symbols_from_file, normalize_symbols
from src.config.settings import load_settings
from src.data.provider_factory import create_market_data_provider


def _date(value: str) -> datetime:
    try:
        parsed = datetime.strptime(value, "%Y-%m-%d")
        return datetime.combine(parsed.date(), time(15, 0))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("日期格式应为 YYYY-MM-DD") from exc


def _print_groups(title: str, groups: dict) -> None:
    table = Table(title=title)
    for column in ("group", "count", "hit_5d", "hit_10d", "avg_return_5d", "avg_return_10d", "drawdown_5d"):
        table.add_column(column)
    for name, values in groups.items():
        table.add_row(
            name,
            str(values.get("count", 0)),
            _pct(values.get("hit_rate_5d")),
            _pct(values.get("hit_rate_10d")),
            _pct(values.get("avg_return_5d")),
            _pct(values.get("avg_return_10d")),
            _pct(values.get("avg_max_drawdown_5d")),
        )
    Console().print(table)


def _pct(value) -> str:
    return "-" if value is None else f"{float(value):.2%}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run isolated StockLens Algorithm Audit Lab.")
    parser.add_argument("--start", required=True, type=_date)
    parser.add_argument("--end", required=True, type=_date)
    parser.add_argument("--symbols", nargs="*", default=[])
    parser.add_argument("--symbols-file")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--step-days", type=int, default=5)
    parser.add_argument("--lookback-days", type=int, default=120)
    parser.add_argument("--max-symbols", type=int)
    parser.add_argument("--audit-name")
    parser.add_argument("--export-csv", action="store_true")
    args = parser.parse_args()

    raw_symbols = list(args.symbols)
    if args.symbols_file:
        raw_symbols.extend(load_symbols_from_file(args.symbols_file))
    if args.demo and not raw_symbols:
        raw_symbols = list(SMALL_DEMO_UNIVERSE)
    try:
        symbols = normalize_symbols(raw_symbols)
        request = AuditRequest(
            start_date=args.start,
            end_date=args.end,
            symbols=symbols,
            step_days=args.step_days,
            lookback_days=args.lookback_days,
            max_symbols=args.max_symbols,
            audit_name=args.audit_name,
        )
    except Exception as exc:
        print(f"审查参数错误：{exc}")
        return

    settings = load_settings()
    provider = create_market_data_provider(
        settings.market_provider,
        cache_dir="data/audit/cache",
        use_cache=True,
    )
    store = AuditStore()
    audit_id = store.create_run(request)
    try:
        samples = AlgorithmAuditRunner(settings, provider).run(request, audit_id=audit_id)
        store.save_samples(samples)
        summary = AuditMetricsBuilder.build_summary(samples)
        store.finalize_run(audit_id, summary)
    except Exception as exc:
        print(f"算法审查失败：{type(exc).__name__}: {exc}")
        return

    print("\nStockLens Algorithm Audit Lab（独立实验，不影响正式数据）")
    print(f"audit_id: {audit_id}")
    print(f"samples_count: {summary['samples_count']}")
    print(f"complete_samples: {summary['complete_samples']}")
    print(f"action_distribution: {summary['action_distribution']}")
    print(f"quant_score corr future_return_5d: {summary['score_future_return_corr_5d']}")
    print(f"quant_score corr future_return_10d: {summary['score_future_return_corr_10d']}")
    print(f"final_score corr future_return_5d: {summary['final_score_future_return_corr_5d']}")
    print(f"final_score corr future_return_10d: {summary['final_score_future_return_corr_10d']}")
    print(f"ranking_warning: {summary['ranking_warning']}")
    _print_groups("Action Metrics", summary["action_metrics"])
    _print_groups("Quant Decision Metrics", summary["quant_decision_metrics"])

    if args.export_csv:
        paths = store.export_csv(audit_id)
        print("CSV 导出路径：")
        for name, path in paths.items():
            print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
