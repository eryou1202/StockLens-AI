from __future__ import annotations

import argparse
import json

from rich.console import Console
from rich.table import Table

from src.audit.audit_store import AuditStore


def _pct(value) -> str:
    return "-" if value is None else f"{float(value):.2%}"


def _table(title: str, groups: dict) -> Table:
    table = Table(title=title)
    for column in ("group", "count", "hit_rate_5d", "hit_rate_10d", "avg_return_5d", "avg_return_10d"):
        table.add_column(column)
    for name, values in groups.items():
        table.add_row(
            name,
            str(values.get("count", 0)),
            _pct(values.get("hit_rate_5d")),
            _pct(values.get("hit_rate_10d")),
            _pct(values.get("avg_return_5d")),
            _pct(values.get("avg_return_10d")),
        )
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description="Show one Algorithm Audit summary.")
    parser.add_argument("--audit-id", default="latest")
    args = parser.parse_args()
    try:
        summary = AuditStore().load_summary(args.audit_id)
    except Exception as exc:
        print(f"读取审查摘要失败：{type(exc).__name__}: {exc}")
        return

    print("StockLens Algorithm Audit Summary")
    print(f"audit_id: {summary.audit_id}")
    print(f"audit_name: {summary.audit_name or '-'}")
    print(f"range: {summary.start_date.date()} -> {summary.end_date.date()}")
    print(f"symbols_count: {summary.symbols_count}")
    print(f"samples_count: {summary.samples_count}")
    print(f"complete_samples: {summary.complete_samples}")
    print("action_distribution: " + json.dumps(summary.action_distribution, ensure_ascii=False))
    print(f"quant_score corr 5d: {summary.score_future_return_corr_5d}")
    print(f"quant_score corr 10d: {summary.score_future_return_corr_10d}")
    print(f"final_score corr 5d: {summary.final_score_future_return_corr_5d}")
    print(f"final_score corr 10d: {summary.final_score_future_return_corr_10d}")
    print(f"ranking_warning: {summary.ranking_warning}")
    Console().print(_table("Action Metrics", summary.action_metrics))
    Console().print(_table("Quant Decision Metrics", summary.quant_decision_metrics))
    print("注意：独立实验结果不代表正式推荐记录或真实交易表现。")


if __name__ == "__main__":
    main()
