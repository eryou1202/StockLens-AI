from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from src.tracking.recommendation_tracking_analytics import RecommendationTrackingAnalytics


def _fmt_pct(value: Any) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return "-"


def _print_horizon_line(summary: dict[str, Any], horizon: int) -> None:
    progress = summary.get("completion_progress", {})
    overall = summary.get("overall_metrics", {}).get(f"{horizon}d", {})
    print(
        f"{horizon}D: completed={progress.get(f'completed_{horizon}d_count', 0)}, "
        f"pending={progress.get(f'pending_{horizon}d_count', 0)}, "
        f"avg_return={_fmt_pct(overall.get('avg_return'))}, "
        f"positive_rate={_fmt_pct(overall.get('positive_return_rate'))}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build read-only analytics for formal recommendation_tracking records."
    )
    parser.add_argument(
        "--database-path",
        default="data/signals.sqlite",
        help="Path to the formal StockLens SQLite database.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/tracking",
        help="Output directory for tracking analytics report files.",
    )
    args = parser.parse_args()

    analytics = RecommendationTrackingAnalytics(database_path=args.database_path)
    summary = analytics.build_report(output_dir=args.output_dir)
    outputs = summary.get("outputs", {})
    progress = summary.get("completion_progress", {})

    print("StockLens Recommendation Tracking Analytics")
    print(f"database: {Path(args.database_path)}")
    print(f"total_records: {summary.get('total_records', 0)}")
    print(f"tracking_count: {progress.get('tracking_tracking_count', 0)}")
    print(f"complete_count: {progress.get('complete_tracking_count', 0)}")
    print(f"failed_count: {progress.get('failed_tracking_count', 0)}")
    for horizon in (1, 3, 5, 10):
        _print_horizon_line(summary, horizon)
    print(f"missing_columns: {summary.get('missing_columns', [])}")
    print(f"database_unchanged: {summary.get('database_unchanged')}")
    print(f"summary_json: {outputs.get('summary')}")
    print(f"metrics_csv: {outputs.get('metrics')}")
    print(f"cases_csv: {outputs.get('cases')}")
    print("notice: formal tracking analytics are read-only and do not change recommendations.")


if __name__ == "__main__":
    main()
