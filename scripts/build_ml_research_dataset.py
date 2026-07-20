from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.audit.universe_loader import load_symbols_from_file, normalize_symbols
from src.config.settings import load_settings
from src.data.provider_factory import create_market_data_provider
from src.ml.ml_dataset_builder import MLDatasetBuilder
from src.ml.ml_dataset_store import MLDatasetStore
from src.ml.ml_schema import MLDatasetRequest


LABEL_PREFIXES = (
    "future_return_",
    "future_excess_return_",
    "future_rank_pct_",
    "future_top30_",
    "future_bottom30_",
    "hit_",
    "future_max_drawdown_",
)
NON_FEATURE_COLUMNS = {
    "sample_id",
    "symbol",
    "stock_name",
    "as_of_date",
    "price_time",
    "current_price",
    "source",
    "sample_interval_days",
    "lookback_days",
    "label_status",
    "label_error",
}
COMPATIBILITY_KEYS = (
    "symbols_hash",
    "sample_interval_days",
    "lookback_days",
    "horizons",
    "include_context_features",
)
DEFAULT_INCREMENTAL_FULL_FALLBACK_START = datetime(2025, 1, 1)
DATASET_READ_KWARGS = {
    "encoding": "utf-8-sig",
    "low_memory": False,
    "dtype": {"label_error": "string"},
}


def _date(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d")


def _metadata_path(dataset_path: str | Path) -> Path:
    return Path(str(dataset_path) + ".meta.json")


def _read_dataset_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(Path(path), **DATASET_READ_KWARGS)


def _symbols_hash(symbols: list[str]) -> str:
    payload = "\n".join(symbols).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_symbols(args: argparse.Namespace) -> list[str]:
    symbols = list(args.symbols)
    if args.symbols_file:
        symbols.extend(load_symbols_from_file(args.symbols_file))
    symbols = normalize_symbols(symbols)
    return symbols[: args.max_symbols] if args.max_symbols else symbols


def _normalize_as_of_date(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if "as_of_date" in result.columns:
        dates = pd.to_datetime(result["as_of_date"], errors="coerce")
        result["as_of_date"] = dates.dt.strftime("%Y-%m-%d")
    return result


def _label_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if column.startswith(LABEL_PREFIXES)]


def _feature_columns(frame: pd.DataFrame) -> list[str]:
    return [
        column
        for column in frame.columns
        if column not in NON_FEATURE_COLUMNS and not column.startswith(LABEL_PREFIXES)
    ]


def _build_metadata(
    frame: pd.DataFrame,
    args: argparse.Namespace,
    symbols: list[str],
    build_mode: str,
) -> dict[str, Any]:
    dates = pd.to_datetime(frame.get("as_of_date"), errors="coerce")
    return {
        "symbols_file": args.symbols_file,
        "symbols_count": len(symbols),
        "symbols_hash": _symbols_hash(symbols),
        "sample_interval_days": int(args.sample_interval_days),
        "lookback_days": int(args.lookback_days),
        "horizons": sorted(int(item) for item in args.horizons),
        "include_context_features": bool(args.include_context_features),
        "feature_columns": _feature_columns(frame),
        "label_columns": _label_columns(frame),
        "build_mode": build_mode,
        "last_build_time": datetime.now(timezone.utc).isoformat(),
        "min_as_of_date": (
            dates.min().date().isoformat() if dates.notna().any() else None
        ),
        "max_as_of_date": (
            dates.max().date().isoformat() if dates.notna().any() else None
        ),
    }


def _expected_compatibility(args: argparse.Namespace, symbols: list[str]) -> dict[str, Any]:
    return {
        "symbols_hash": _symbols_hash(symbols),
        "sample_interval_days": int(args.sample_interval_days),
        "lookback_days": int(args.lookback_days),
        "horizons": sorted(int(item) for item in args.horizons),
        "include_context_features": bool(args.include_context_features),
    }


def _check_incremental_metadata(
    metadata_file: Path,
    args: argparse.Namespace,
    symbols: list[str],
) -> None:
    if not metadata_file.exists():
        print(
            f"warning: metadata sidecar not found: {metadata_file}; "
            "allowing first compatible incremental run and creating metadata after save."
        )
        return

    existing = json.loads(metadata_file.read_text(encoding="utf-8"))
    expected = _expected_compatibility(args, symbols)
    mismatches: list[str] = []
    for key in COMPATIBILITY_KEYS:
        if existing.get(key) != expected[key]:
            mismatches.append(
                f"{key}: existing={existing.get(key)!r}, requested={expected[key]!r}"
            )
    if mismatches:
        joined = "\n  ".join(mismatches)
        raise RuntimeError(
            "incremental update refused because dataset metadata is incompatible; "
            "run a full rebuild instead.\n  "
            + joined
        )


def _load_existing_dataset(path: Path) -> pd.DataFrame:
    frame = _read_dataset_csv(path)
    required = {"as_of_date", "symbol"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise RuntimeError(f"existing dataset missing required columns: {missing}")
    frame = _normalize_as_of_date(frame)
    if pd.to_datetime(frame["as_of_date"], errors="coerce").isna().all():
        raise RuntimeError("existing dataset has no valid as_of_date values")
    return frame


def _build_frame(request: MLDatasetRequest) -> pd.DataFrame:
    samples = MLDatasetBuilder(load_settings()).build(request)
    return MLDatasetStore().to_frame(samples)


def _build_frame_with_cache_mode(request: MLDatasetRequest, *, use_cache: bool) -> pd.DataFrame:
    samples = MLDatasetBuilder(load_settings(), use_cache=use_cache).build(request)
    return MLDatasetStore().to_frame(samples)


def _deduplicate_and_sort(frame: pd.DataFrame) -> pd.DataFrame:
    result = _normalize_as_of_date(frame)
    result = result.drop_duplicates(["as_of_date", "symbol"], keep="last")
    return result.sort_values(["as_of_date", "symbol"], kind="stable").reset_index(drop=True)


def _duplicate_key_count(frame: pd.DataFrame) -> int:
    if {"as_of_date", "symbol"}.issubset(frame.columns):
        return int(frame.duplicated(["as_of_date", "symbol"]).sum())
    return 0


def _date_range_from_frame(frame: pd.DataFrame, column: str) -> tuple[str, str]:
    if frame.empty or column not in frame.columns:
        return "-", "-"
    dates = pd.to_datetime(frame[column], errors="coerce")
    if not dates.notna().any():
        return "-", "-"
    return dates.min().date().isoformat(), dates.max().date().isoformat()


def _market_date_probe(
    symbols: list[str],
    start_time: datetime,
    end_time: datetime,
    max_symbols: int = 5,
) -> dict[str, Any]:
    settings = load_settings()
    provider = create_market_data_provider(
        settings.market_provider,
        cache_dir="data/ml/cache",
        use_cache=False,
    )
    dates: set[datetime.date] = set()
    errors: dict[str, str] = {}
    probed_symbols = symbols[: min(len(symbols), max_symbols)]
    with provider.session():
        for symbol in probed_symbols:
            try:
                bundle = provider.get_bars(
                    symbol=symbol,
                    start_time=start_time,
                    end_time=end_time,
                    frequency=settings.market_frequency,
                    adjust_type=settings.market_adjust_type,
                )
            except Exception as exc:
                errors[symbol] = f"{type(exc).__name__}: {exc}"
                continue
            for bar in bundle.sorted_bars():
                if bar.trade_time is None:
                    continue
                if bar.trade_status is not None and str(bar.trade_status).strip() != "1":
                    continue
                trade_date = bar.trade_time.replace(tzinfo=None).date()
                if start_time.date() <= trade_date <= end_time.date():
                    dates.add(trade_date)
    sorted_dates = sorted(dates)
    return {
        "symbols": probed_symbols,
        "raw_market_min_date": sorted_dates[0].isoformat() if sorted_dates else "-",
        "raw_market_max_date": sorted_dates[-1].isoformat() if sorted_dates else "-",
        "raw_market_dates_after_existing": sorted_dates,
        "errors": errors,
    }


def _validate_written_dataset(
    tmp_path: Path,
    requested_end_date: datetime,
) -> pd.DataFrame:
    frame = _read_dataset_csv(tmp_path)
    if frame.empty:
        raise RuntimeError("validation failed: output dataset is empty")
    required = {"as_of_date", "symbol"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise RuntimeError(f"validation failed: missing required columns {missing}")
    duplicate_count = _duplicate_key_count(frame)
    if duplicate_count:
        raise RuntimeError(
            f"validation failed: duplicate as_of_date + symbol keys = {duplicate_count}"
        )
    dates = pd.to_datetime(frame["as_of_date"], errors="coerce")
    if dates.isna().all():
        raise RuntimeError("validation failed: no valid as_of_date values")
    if dates.max().date() > requested_end_date.date():
        raise RuntimeError(
            "validation failed: max as_of_date is later than requested end date "
            f"({dates.max().date()} > {requested_end_date.date()})"
        )
    return frame


def _write_atomic(
    frame: pd.DataFrame,
    output_path: str,
    metadata: dict[str, Any],
    requested_end_date: datetime,
) -> pd.DataFrame:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp.csv")
    metadata_path = _metadata_path(path)
    metadata_tmp_path = metadata_path.with_name(metadata_path.name + ".tmp")

    frame.to_csv(tmp_path, index=False, encoding="utf-8-sig")
    validated = _validate_written_dataset(tmp_path, requested_end_date)

    metadata_tmp_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(tmp_path, path)
    os.replace(metadata_tmp_path, metadata_path)
    return validated


def _trading_dates_from_provider(
    symbols: list[str],
    existing_max_date: datetime,
    refresh_trading_days: int,
) -> list[datetime]:
    settings = load_settings()
    provider = create_market_data_provider(
        settings.market_provider,
        cache_dir="data/ml/cache",
        use_cache=True,
    )
    collected_dates: set[datetime.date] = set()
    end_time = existing_max_date
    window_days = max(refresh_trading_days * 4 + 45, 75)
    candidate_symbols = symbols[: min(len(symbols), 30)]

    with provider.session():
        for attempt in range(3):
            start_time = existing_max_date - timedelta(days=window_days * (attempt + 1))
            for symbol in candidate_symbols:
                try:
                    bundle = provider.get_bars(
                        symbol=symbol,
                        start_time=start_time,
                        end_time=end_time,
                        frequency=settings.market_frequency,
                        adjust_type=settings.market_adjust_type,
                    )
                except Exception:
                    continue
                for bar in bundle.sorted_bars():
                    if bar.trade_time is None:
                        continue
                    trade_time = bar.trade_time.replace(tzinfo=None)
                    trade_date = trade_time.date()
                    if not start_time.date() <= trade_date <= existing_max_date.date():
                        continue
                    if bar.trade_status is not None and str(bar.trade_status).strip() != "1":
                        continue
                    collected_dates.add(trade_date)
            if len(collected_dates) >= refresh_trading_days:
                break

    return [
        datetime.combine(date, existing_max_date.time())
        for date in sorted(collected_dates)
        if date <= existing_max_date.date()
    ]


def _refresh_start_from_trading_days(
    existing_frame: pd.DataFrame,
    symbols: list[str],
    existing_max_date: datetime,
    refresh_trading_days: int,
) -> datetime:
    trading_dates = _trading_dates_from_provider(
        symbols,
        existing_max_date,
        refresh_trading_days,
    )
    if len(trading_dates) < refresh_trading_days:
        existing_dates = sorted(
            pd.to_datetime(existing_frame["as_of_date"], errors="coerce")
            .dropna()
            .dt.date
            .unique()
        )
        trading_dates = [
            datetime.combine(date, existing_max_date.time())
            for date in existing_dates
            if date <= existing_max_date.date()
        ]
        print(
            "warning: provider trading-date boundary lookup was incomplete; "
            "falling back to existing dataset as_of_date sequence."
        )
    if not trading_dates:
        raise RuntimeError("unable to determine refresh_start_date from trading dates")
    refresh_days = max(1, int(refresh_trading_days))
    return trading_dates[-refresh_days] if len(trading_dates) >= refresh_days else trading_dates[0]


def _make_request(
    args: argparse.Namespace,
    symbols: list[str],
    start_date: datetime,
    end_date: datetime,
) -> MLDatasetRequest:
    return MLDatasetRequest(
        start_date=start_date,
        end_date=end_date,
        symbols=symbols,
        max_symbols=None,
        lookback_days=args.lookback_days,
        sample_interval_days=args.sample_interval_days,
        horizons=args.horizons,
        include_context_features=args.include_context_features,
        output_path=args.output,
    )


def _print_label_availability(frame: pd.DataFrame, horizons: list[int]) -> None:
    for horizon in sorted(set(horizons)):
        column = f"future_return_{horizon}d"
        available = (
            int(pd.to_numeric(frame[column], errors="coerce").notna().sum())
            if column in frame.columns else 0
        )
        print(f"{column}_available: {available}")


def _print_common_output(frame: pd.DataFrame, output: str) -> None:
    dates = pd.to_datetime(frame["as_of_date"], errors="coerce")
    status = (
        frame["label_status"].value_counts(dropna=False).to_dict()
        if "label_status" in frame.columns else {}
    )
    print(f"rows: {len(frame)}")
    print(f"symbols: {frame['symbol'].nunique() if not frame.empty else 0}")
    print(f"label_status: {status}")
    print(
        "date_range: "
        f"{dates.min().date() if dates.notna().any() else '-'} -> "
        f"{dates.max().date() if dates.notna().any() else '-'}"
    )
    print(f"output: {output}")
    print(f"metadata_output: {_metadata_path(output)}")


def _run_full_build(
    args: argparse.Namespace,
    symbols: list[str],
    build_mode: str = "full",
) -> pd.DataFrame:
    if args.start is None:
        raise RuntimeError("--start is required for a full dataset build")
    request = _make_request(args, symbols, args.start, args.end)
    frame = _deduplicate_and_sort(_build_frame(request))
    metadata = _build_metadata(frame, args, symbols, build_mode)
    written = _write_atomic(frame, args.output, metadata, args.end)
    print(f"build_mode: {build_mode}")
    _print_common_output(written, args.output)
    _print_label_availability(written, args.horizons)
    return written


def _run_incremental_build(args: argparse.Namespace, symbols: list[str]) -> pd.DataFrame:
    existing_path = Path(args.existing_dataset or args.output)
    if not existing_path.exists():
        print(
            f"existing dataset not found: {existing_path}; "
            "falling back to full build."
        )
        if args.start is None:
            args.start = DEFAULT_INCREMENTAL_FULL_FALLBACK_START
            print(
                "warning: --start was not provided; full fallback uses "
                f"{args.start.date().isoformat()}. Pass --start to override."
            )
        return _run_full_build(args, symbols, build_mode="full_fallback")

    metadata_file = _metadata_path(existing_path)
    _check_incremental_metadata(metadata_file, args, symbols)

    existing = _load_existing_dataset(existing_path)
    existing_dates = pd.to_datetime(existing["as_of_date"], errors="coerce")
    existing_min_date = existing_dates.min()
    existing_max_date = existing_dates.max()
    refresh_start_date = _refresh_start_from_trading_days(
        existing,
        symbols,
        existing_max_date.to_pydatetime(),
        args.refresh_trading_days,
    )
    preserved = existing.loc[existing_dates.dt.date < refresh_start_date.date()].copy()

    max_horizon = max(args.horizons) if args.horizons else 1
    data_fetch_start = refresh_start_date - timedelta(days=args.lookback_days + 10)
    data_fetch_end = args.end + timedelta(days=max_horizon * 3 + 15)
    market_probe = _market_date_probe(symbols, data_fetch_start, data_fetch_end)
    provider_new_dates = [
        item
        for item in market_probe["raw_market_dates_after_existing"]
        if existing_max_date.date() < item <= args.end.date()
    ]

    request = _make_request(args, symbols, refresh_start_date, args.end)
    refreshed_tail = _normalize_as_of_date(
        _build_frame_with_cache_mode(request, use_cache=False)
    )
    refreshed_tail_dates = pd.to_datetime(
        refreshed_tail.get("as_of_date"), errors="coerce"
    )
    refreshed_tail_max = (
        refreshed_tail_dates.max().date() if refreshed_tail_dates.notna().any() else None
    )
    generated_sample_min_date, generated_sample_max_date = _date_range_from_frame(
        refreshed_tail,
        "as_of_date",
    )
    refreshed_tail_min_date, refreshed_tail_max_date = generated_sample_min_date, generated_sample_max_date
    refreshed_dates_set = set(refreshed_tail_dates.dt.date.dropna())
    existing_dates_set = set(existing_dates.dt.date.dropna())
    new_dates = sorted(refreshed_dates_set - existing_dates_set)

    print(f"data_fetch_start: {data_fetch_start.date()}")
    print(f"data_fetch_end: {data_fetch_end.date()}")
    print(f"raw_market_min_date: {market_probe['raw_market_min_date']}")
    print(f"raw_market_max_date: {market_probe['raw_market_max_date']}")
    print(f"generated_sample_min_date: {generated_sample_min_date}")
    print(f"generated_sample_max_date: {generated_sample_max_date}")
    print(f"refreshed_tail_min_date: {refreshed_tail_min_date}")
    print(f"refreshed_tail_max_date: {refreshed_tail_max_date}")
    print(f"existing_max_date: {existing_max_date.date()}")
    print(f"new_dates: {[item.isoformat() for item in new_dates]}")
    if market_probe["errors"]:
        print(f"market_probe_errors: {market_probe['errors']}")

    if provider_new_dates and (
        refreshed_tail_max is None or refreshed_tail_max <= existing_max_date.date()
    ):
        raise RuntimeError(
            "incremental update refused: provider no-cache probe has dates after "
            f"existing_max_date ({[item.isoformat() for item in provider_new_dates]}), "
            "but refreshed_tail did not generate any newer as_of_date. "
            "The existing dataset was left unchanged."
        )

    final = _deduplicate_and_sort(
        pd.concat([preserved, refreshed_tail], ignore_index=True, sort=False)
    )
    duplicate_key_count = _duplicate_key_count(final)
    metadata = _build_metadata(final, args, symbols, "incremental")
    written = _write_atomic(final, args.output, metadata, args.end)

    final_dates = pd.to_datetime(written["as_of_date"], errors="coerce")
    new_dates_count = len(new_dates)
    print("build_mode: incremental")
    print(f"existing_rows: {len(existing)}")
    print(f"existing_min_date: {existing_min_date.date()}")
    print(f"existing_max_date: {existing_max_date.date()}")
    print(f"refresh_start_date: {refresh_start_date.date()}")
    print(f"requested_end_date: {args.end.date()}")
    print(f"preserved_rows: {len(preserved)}")
    print(f"refreshed_rows: {len(refreshed_tail)}")
    print(f"new_dates_count: {new_dates_count}")
    print(f"final_rows: {len(written)}")
    print(f"final_min_date: {final_dates.min().date() if final_dates.notna().any() else '-'}")
    print(f"final_max_date: {final_dates.max().date() if final_dates.notna().any() else '-'}")
    print(f"duplicate_key_count: {duplicate_key_count}")
    print(f"output: {args.output}")
    print(f"metadata_output: {_metadata_path(args.output)}")
    _print_label_availability(written, args.horizons)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="构建隔离的多周期 ML 研究数据集")
    parser.add_argument("--start", type=_date)
    parser.add_argument("--end", required=True, type=_date)
    parser.add_argument("--symbols", nargs="*", default=[])
    parser.add_argument("--symbols-file")
    parser.add_argument("--max-symbols", type=int)
    parser.add_argument("--lookback-days", type=int, default=120)
    parser.add_argument("--sample-interval-days", type=int, default=5)
    parser.add_argument("--horizons", nargs="+", type=int, default=[1, 2, 3, 5, 10, 20])
    parser.add_argument(
        "--include-context-features",
        action="store_true",
        help="加入仅使用 as_of_date 当日及以前数据的市场/宽度/相对强弱/风格特征",
    )
    parser.add_argument("--incremental", action="store_true")
    parser.add_argument("--refresh-trading-days", type=int, default=30)
    parser.add_argument("--existing-dataset")
    parser.add_argument("--output", default="data/ml/ml_research_dataset.csv")
    args = parser.parse_args()

    symbols = _load_symbols(args)
    if not symbols:
        parser.error("请通过 --symbols 或 --symbols-file 提供至少一只股票")
    if args.start is not None and args.start > args.end:
        parser.error("--start 不能晚于 --end")
    if args.refresh_trading_days < 1:
        parser.error("--refresh-trading-days must be >= 1")

    print("StockLens Multi-Horizon ML Research Dataset")
    print("特征严格只使用 as_of_date 当日及之前行情；未来行情仅用于标签。")
    print("sample_interval_days is interpreted as trading-bar interval.")
    print(f"context features: {'enabled' if args.include_context_features else 'disabled'}")

    try:
        if args.incremental:
            _run_incremental_build(args, symbols)
        else:
            _run_full_build(args, symbols)
    except RuntimeError as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()
