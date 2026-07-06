from __future__ import annotations

import argparse
from rich.console import Console
from rich.table import Table
from src.config.settings import load_settings
from src.tracking.recommendation_tracker import RecommendationTracker


def _pct(value: float | None) -> str:
    return "-" if value is None else f"{value:.2%}"


def main() -> None:
    parser = argparse.ArgumentParser(description="List recommendation tracking snapshots.")
    parser.add_argument("--status", choices=["all", "tracking", "complete", "failed"], default="all")
    args = parser.parse_args()
    items = RecommendationTracker(load_settings().database_path).list_tracking(args.status)
    if not items:
        print(f"没有 status={args.status} 的追踪记录。可运行 scripts.run_recommendations --save-tracking。")
        return
    table = Table(title="StockLens Recommendation Tracking")
    for name in ("id", "symbol", "as_of_time", "source", "action", "score", "status", "return_5d", "return_10d", "verdict"):
        table.add_column(name)
    for item in items:
        table.add_row(
            str(item.id), item.symbol, item.as_of_time.isoformat(), item.source_type or "-",
            item.action, "-" if item.final_score is None else f"{item.final_score:.2f}",
            item.tracking_status.value, _pct(item.future_return_5d), _pct(item.future_return_10d),
            item.manual_verdict.value if item.manual_verdict else "-",
        )
    Console().print(table)


if __name__ == "__main__":
    main()
