from __future__ import annotations

import argparse
from datetime import date, datetime, time, timedelta

from src.ai.file_ai_info_engine import FileAIInfoEngine
from src.config.settings import load_settings
from src.data.provider_factory import create_market_data_provider
from src.decision.decision_engine import DecisionEngine
from src.pipeline import ResearchPipeline
from src.quant.quant_engine import RuleBasedQuantEngine
from src.reports.report_builder import ReportBuilder
from src.storage.sqlite_store import SQLiteSignalStore


def _parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("日期格式应为 YYYY-MM-DD") from exc


def _positive_int(value: str) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--step-days 必须是正整数") from exc
    if result <= 0:
        raise argparse.ArgumentTypeError("--step-days 必须大于 0")
    return result


def _date_range(start: date, end: date, step_days: int) -> list[date]:
    values: list[date] = []
    current = start
    while current <= end:
        values.append(current)
        current += timedelta(days=step_days)
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate StockLens signals over historical dates.")
    parser.add_argument("--start", required=True, type=_parse_date)
    parser.add_argument("--end", required=True, type=_parse_date)
    parser.add_argument("--step-days", type=_positive_int, default=3)
    args = parser.parse_args()
    if args.start > args.end:
        parser.error("--start 不能晚于 --end")

    settings = load_settings()
    report_builder = ReportBuilder()
    # 批量历史区间必须读取每个 as-of 的完整范围，避免截断缓存污染后续日期。
    market_provider = create_market_data_provider(
        name=settings.market_provider,
        cache_dir=settings.cache_dir,
        use_cache=False,
    )
    pipeline = ResearchPipeline(
        ai_engine=FileAIInfoEngine("data/ai_candidates.json"),
        market_data_provider=market_provider,
        quant_engine=RuleBasedQuantEngine(),
        decision_engine=DecisionEngine(),
        store=SQLiteSignalStore(settings.database_path),
        report_builder=report_builder,
        lookback_days=settings.default_lookback_days,
        frequency=settings.market_frequency,
        adjust_type=settings.market_adjust_type,
    )

    dates = _date_range(args.start, args.end, args.step_days)
    total_decisions = 0
    failed_dates: list[str] = []
    print("StockLens Historical Batch v1.0")
    print(f"database: {settings.database_path}")
    print(f"provider: {market_provider.provider_name}\n")

    for current_date in dates:
        as_of_time = datetime.combine(current_date, time(15, 0, 0))
        try:
            decisions = pipeline.run_once(as_of_time)
            generated_count = len(decisions)
            total_decisions += generated_count
            print(f"as_of_time: {as_of_time.isoformat()}")
            print(f"generated decisions count: {generated_count}")
            print(f"database path: {settings.database_path}\n")
        except Exception as exc:
            failed_dates.append(current_date.isoformat())
            print(f"as_of_time: {as_of_time.isoformat()}")
            print("generated decisions count: 0")
            print(f"database path: {settings.database_path}")
            print(f"error: {type(exc).__name__}: {exc}\n")

    print("Historical Batch Summary")
    print(f"total_dates: {len(dates)}")
    print(f"total_decisions: {total_decisions}")
    print(f"failed_dates: {len(failed_dates)}")
    if failed_dates:
        print(f"failed_date_values: {', '.join(failed_dates)}")


if __name__ == "__main__":
    main()
