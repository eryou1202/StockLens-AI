from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


HORIZONS = (1, 3, 5, 10)
DEFAULT_DATES = ("2026-07-07", "2026-07-09", "2026-07-13")
SCOPE_BUCKETS = {
    "top10": {"top10"},
    "top20": {"top10", "top20"},
    "top50": {"top10", "top20", "top50"},
    "outside_top50": {"outside_top50"},
}
FEATURE_COLUMNS = (
    "ml_score",
    "quant_score",
    "risk_score",
    "overheat_score",
    "return_5d",
    "return_20d",
    "volume_ratio_5d",
    "breadth_up_ratio_5d",
    "breadth_above_ma20_ratio",
)
RISK_TAGS = (
    "extreme_risk_score",
    "high_risk_score",
    "recent_5d_drop",
    "extreme_5d_drop",
    "high_volatility_reversal",
    "quant_disagree",
    "strong_quant_disagree",
    "weak_market_breadth",
    "weak_ma_breadth",
    "overheat",
    "rebound_candidate",
)
BASE_COLUMNS = {
    "as_of_date",
    "symbol",
    "stock_name",
    "ml_rank",
    "ml_bucket",
    "ml_score",
    "quant_score",
    "risk_score",
    "overheat_score",
    "return_5d",
    "return_20d",
    "volume_ratio_5d",
    "breadth_up_ratio_5d",
    "breadth_above_ma20_ratio",
    "shadow_risk_level",
    "shadow_risk_tags",
    "shadow_action",
    "future_return_1d",
    "future_return_3d",
    "future_return_5d",
    "future_return_10d",
    "future_excess_return_5d",
    "future_rank_pct_5d",
    "hit_5d",
    "outcome_status",
}


def _date(value: str) -> str:
    return pd.Timestamp(value).date().isoformat()


def _safe_read_csv(path: Path, usecols: set[str] | None = None) -> pd.DataFrame:
    header = pd.read_csv(path, encoding="utf-8-sig", nrows=0)
    columns = set(header.columns)
    selected = columns if usecols is None else columns.intersection(usecols)
    dtype: dict[str, str] = {}
    if "label_error" in selected:
        dtype["label_error"] = "string"
    return pd.read_csv(
        path,
        encoding="utf-8-sig",
        low_memory=False,
        dtype=dtype or None,
        usecols=lambda column: column in selected,
    )


def _number_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce")


def _mean(frame: pd.DataFrame, column: str) -> float | None:
    value = _number_series(frame, column).mean()
    return None if pd.isna(value) else float(value)


def _median(frame: pd.DataFrame, column: str) -> float | None:
    value = _number_series(frame, column).median()
    return None if pd.isna(value) else float(value)


def _completed(frame: pd.DataFrame, horizon: int) -> pd.Series:
    return _number_series(frame, f"future_return_{horizon}d").notna()


def _scope_frame(frame: pd.DataFrame, scope: str) -> pd.DataFrame:
    return frame.loc[frame["ml_bucket"].isin(SCOPE_BUCKETS[scope])].copy()


def _hit_rate(frame: pd.DataFrame, horizon: int) -> float | None:
    hit_column = f"hit_{horizon}d"
    if hit_column in frame.columns:
        values = _number_series(frame, hit_column)
    else:
        returns = _number_series(frame, f"future_return_{horizon}d")
        values = returns.gt(0).where(returns.notna())
    value = values.mean()
    return None if pd.isna(value) else float(value)


def _scope_metrics(
    frame: pd.DataFrame,
    scope: str,
    primary_horizon: int,
    outside_mean_by_horizon: dict[int, float | None],
    all_mean_by_horizon: dict[int, float | None],
) -> dict[str, Any]:
    group = _scope_frame(frame, scope)
    result: dict[str, Any] = {
        "sample_count": int(len(group)),
        "completed_count": int(_completed(group, primary_horizon).sum()),
    }
    for horizon in HORIZONS:
        return_column = f"future_return_{horizon}d"
        avg = _mean(group, return_column)
        result[f"avg_return_{horizon}d"] = avg
        result[f"completed_{horizon}d_count"] = int(_completed(group, horizon).sum())
        result[f"hit_rate_{horizon}d"] = _hit_rate(group, horizon)
        all_avg = all_mean_by_horizon.get(horizon)
        outside_avg = outside_mean_by_horizon.get(horizon)
        result[f"excess_vs_all_{horizon}d"] = (
            None if avg is None or all_avg is None else avg - all_avg
        )
        result[f"excess_vs_outside_top50_{horizon}d"] = (
            None if avg is None or outside_avg is None else avg - outside_avg
        )
    result[f"avg_future_rank_pct_{primary_horizon}d"] = _mean(
        group, f"future_rank_pct_{primary_horizon}d"
    )
    result[f"avg_future_excess_return_{primary_horizon}d"] = _mean(
        group, f"future_excess_return_{primary_horizon}d"
    )
    return result


def _parse_tags(value: Any) -> set[str]:
    if value is None or pd.isna(value):
        return set()
    return {item.strip() for item in str(value).split(";") if item.strip()}


def _quant_bucket(value: Any) -> str:
    try:
        score = float(value)
    except (TypeError, ValueError, OverflowError):
        return "quant_missing"
    if not np.isfinite(score):
        return "quant_missing"
    if score < 30:
        return "quant_lt_30"
    if score < 50:
        return "quant_30_50"
    if score < 70:
        return "quant_50_70"
    return "quant_ge_70"


def _board(symbol: Any) -> str:
    text = str(symbol or "").strip().upper()
    code = text.split(".")[0]
    exchange = text.split(".")[1] if "." in text else ""
    if exchange == "BJ" or code.startswith(("4", "8", "9")):
        return "beijing"
    if code.startswith(("300", "301")):
        return "chinext"
    if code.startswith(("688", "689")):
        return "star"
    if exchange in {"SH", "SZ"}:
        return "main_board"
    return "unknown"


def _add_row(
    rows: list[dict[str, Any]],
    analysis_type: str,
    as_of_date: str | None,
    comparison_date: str | None,
    scope: str | None,
    group: str | None,
    metric: str,
    value: Any,
    sample_count: int | None = None,
    completed_count: int | None = None,
    note: str | None = None,
) -> None:
    rows.append({
        "analysis_type": analysis_type,
        "as_of_date": as_of_date,
        "comparison_date": comparison_date,
        "scope": scope,
        "group": group,
        "metric": metric,
        "value": value,
        "sample_count": sample_count,
        "completed_count": completed_count,
        "note": note,
    })


def _clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _clean_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clean_json(item) for item in value]
    if isinstance(value, tuple):
        return [_clean_json(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return None if not np.isfinite(value) else value
    if pd.isna(value) if value is not None and not isinstance(value, (str, bytes)) else False:
        return None
    return value


def _load_inputs(outcomes_file: Path, history_file: Path) -> pd.DataFrame:
    outcomes = _safe_read_csv(outcomes_file, BASE_COLUMNS)
    history = _safe_read_csv(history_file, BASE_COLUMNS)
    for frame in (outcomes, history):
        frame["as_of_date"] = pd.to_datetime(
            frame["as_of_date"], errors="coerce"
        ).dt.strftime("%Y-%m-%d")
        frame["symbol"] = frame["symbol"].astype("string").str.strip().str.upper()

    key = ["as_of_date", "symbol"]
    outcome_keys = set(map(tuple, outcomes[key].dropna().astype(str).to_numpy()))
    if history.empty:
        combined = outcomes.copy()
    else:
        history_keys = history[key].astype(str).apply(tuple, axis=1)
        pending_only = history.loc[~history_keys.isin(outcome_keys)].copy()
        combined = pd.concat([outcomes, pending_only], ignore_index=True, sort=False)
    for column in BASE_COLUMNS:
        if column not in combined.columns:
            combined[column] = np.nan
    combined = combined.dropna(subset=["as_of_date", "symbol"])
    combined["shadow_risk_tags_set"] = combined["shadow_risk_tags"].apply(_parse_tags)
    combined["quant_bucket"] = combined["quant_score"].apply(_quant_bucket)
    combined["board"] = combined["symbol"].apply(_board)
    return combined


def _date_summaries(
    frame: pd.DataFrame,
    dates: list[str],
    primary_horizon: int,
    csv_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for date in dates:
        day = frame.loc[frame["as_of_date"].eq(date)]
        if day.empty:
            continue
        all_mean = {h: _mean(day, f"future_return_{h}d") for h in HORIZONS}
        outside = _scope_frame(day, "outside_top50")
        outside_mean = {h: _mean(outside, f"future_return_{h}d") for h in HORIZONS}
        summary: dict[str, Any] = {"total_signals": int(len(day))}
        for horizon in HORIZONS:
            summary[f"completed_{horizon}d_count"] = int(_completed(day, horizon).sum())
        scopes: dict[str, Any] = {}
        for scope in SCOPE_BUCKETS:
            metrics = _scope_metrics(day, scope, primary_horizon, outside_mean, all_mean)
            scopes[scope] = metrics
            for metric, value in metrics.items():
                _add_row(
                    csv_rows,
                    "date_summary",
                    date,
                    None,
                    scope,
                    None,
                    metric,
                    value,
                    metrics.get("sample_count"),
                    metrics.get("completed_count"),
                )
        summary["scopes"] = scopes
        result[date] = summary
    return result


def _feature_comparison(
    frame: pd.DataFrame,
    success_date: str,
    failure_date: str,
    csv_rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[str]]:
    missing = [column for column in FEATURE_COLUMNS if column not in frame.columns]
    available = [column for column in FEATURE_COLUMNS if column not in missing]
    result: dict[str, Any] = {}
    for scope in ("top10", "top20", "top50"):
        result[scope] = {}
        success = _scope_frame(frame.loc[frame["as_of_date"].eq(success_date)], scope)
        failure = _scope_frame(frame.loc[frame["as_of_date"].eq(failure_date)], scope)
        for column in available:
            success_mean = _mean(success, column)
            failure_mean = _mean(failure, column)
            success_median = _median(success, column)
            failure_median = _median(failure, column)
            diff = None if success_mean is None or failure_mean is None else success_mean - failure_mean
            result[scope][column] = {
                "success_mean": success_mean,
                "failure_mean": failure_mean,
                "success_minus_failure_mean": diff,
                "success_median": success_median,
                "failure_median": failure_median,
                "success_missing_count": int(success[column].isna().sum()),
                "failure_missing_count": int(failure[column].isna().sum()),
            }
            _add_row(
                csv_rows,
                "success_vs_failure_feature",
                success_date,
                failure_date,
                scope,
                column,
                "success_minus_failure_mean",
                diff,
                int(len(success) + len(failure)),
                None,
            )
    return result, missing


def _tag_analysis(
    frame: pd.DataFrame,
    dates: list[str],
    success_date: str,
    failure_date: str,
    csv_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    ratios: dict[tuple[str, str, str], float | None] = {}
    for date in dates:
        day = frame.loc[frame["as_of_date"].eq(date)]
        outside_mean_5d = _mean(_scope_frame(day, "outside_top50"), "future_return_5d")
        for scope in SCOPE_BUCKETS:
            scoped = _scope_frame(day, scope)
            denominator = len(scoped)
            for tag in RISK_TAGS:
                tagged = scoped.loc[scoped["shadow_risk_tags_set"].apply(lambda tags: tag in tags)]
                count = int(len(tagged))
                completed_count = int(_completed(tagged, 5).sum())
                avg_5d = _mean(tagged, "future_return_5d")
                ratio = None if denominator == 0 else count / denominator
                ratios[(date, scope, tag)] = ratio
                item = {
                    "as_of_date": date,
                    "scope": scope,
                    "tag": tag,
                    "tag_ratio": ratio,
                    "signal_count": count,
                    "completed_count": completed_count,
                    "avg_return_1d": _mean(tagged, "future_return_1d"),
                    "avg_return_3d": _mean(tagged, "future_return_3d"),
                    "avg_return_5d": avg_5d,
                    "excess_vs_outside_5d": (
                        None if avg_5d is None or outside_mean_5d is None else avg_5d - outside_mean_5d
                    ),
                    "hit_rate_5d": _hit_rate(tagged, 5),
                }
                rows.append(item)
                _add_row(
                    csv_rows,
                    "risk_tag_analysis",
                    date,
                    None,
                    scope,
                    tag,
                    "tag_ratio",
                    ratio,
                    count,
                    completed_count,
                )
    comparison: dict[str, Any] = {}
    for scope in ("top10", "top20", "top50"):
        comparison[scope] = {}
        for tag in RISK_TAGS:
            s = ratios.get((success_date, scope, tag))
            f = ratios.get((failure_date, scope, tag))
            comparison[scope][tag] = {
                "success_ratio": s,
                "failure_ratio": f,
                "success_minus_failure_ratio": None if s is None or f is None else s - f,
            }
    return {"rows": rows, "success_vs_failure_tag_ratio": comparison}


def _grouped_performance(
    frame: pd.DataFrame,
    dates: list[str],
    group_column: str,
    groups: list[str],
    csv_rows: list[dict[str, Any]],
    analysis_type: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for date in dates:
        day = frame.loc[frame["as_of_date"].eq(date)]
        outside_mean_5d = _mean(_scope_frame(day, "outside_top50"), "future_return_5d")
        for scope in SCOPE_BUCKETS:
            scoped = _scope_frame(day, scope)
            for group in groups:
                part = scoped.loc[scoped[group_column].fillna("missing").astype(str).eq(group)]
                avg_5d = _mean(part, "future_return_5d")
                item = {
                    "as_of_date": date,
                    "scope": scope,
                    "group": group,
                    "count": int(len(part)),
                    "completed_5d_count": int(_completed(part, 5).sum()),
                    "avg_ml_score": _mean(part, "ml_score"),
                    "avg_return_1d": _mean(part, "future_return_1d"),
                    "avg_return_3d": _mean(part, "future_return_3d"),
                    "avg_return_5d": avg_5d,
                    "excess_vs_outside_5d": (
                        None if avg_5d is None or outside_mean_5d is None else avg_5d - outside_mean_5d
                    ),
                    "hit_rate_5d": _hit_rate(part, 5),
                }
                rows.append(item)
                _add_row(
                    csv_rows,
                    analysis_type,
                    date,
                    None,
                    scope,
                    group,
                    "count",
                    item["count"],
                    item["count"],
                    item["completed_5d_count"],
                )
    return rows


def _board_concentration(
    frame: pd.DataFrame,
    dates: list[str],
    csv_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for date in dates:
        day = frame.loc[frame["as_of_date"].eq(date)]
        for scope in SCOPE_BUCKETS:
            scoped = _scope_frame(day, scope)
            counts = scoped["board"].fillna("unknown").astype(str).value_counts()
            total = int(len(scoped))
            for board, count in counts.items():
                ratio = None if total == 0 else int(count) / total
                item = {
                    "as_of_date": date,
                    "scope": scope,
                    "board": board,
                    "count": int(count),
                    "ratio": ratio,
                }
                rows.append(item)
                _add_row(
                    csv_rows,
                    "board_concentration",
                    date,
                    None,
                    scope,
                    board,
                    "ratio",
                    ratio,
                    total,
                    None,
                )
    return rows


def _ranking_overlap(
    frame: pd.DataFrame,
    dates: list[str],
    csv_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    existing_dates = [date for date in dates if not frame.loc[frame["as_of_date"].eq(date)].empty]
    rows: list[dict[str, Any]] = []
    for left, right in zip(existing_dates, existing_dates[1:]):
        left_day = frame.loc[frame["as_of_date"].eq(left)]
        right_day = frame.loc[frame["as_of_date"].eq(right)]
        for scope in ("top10", "top20", "top50"):
            left_set = set(_scope_frame(left_day, scope)["symbol"].dropna().astype(str))
            right_set = set(_scope_frame(right_day, scope)["symbol"].dropna().astype(str))
            overlap = left_set.intersection(right_set)
            base = len(left_set) or len(right_set)
            rate = None if base == 0 else len(overlap) / base
            item = {
                "as_of_date": left,
                "comparison_date": right,
                "scope": scope,
                "overlap_count": int(len(overlap)),
                "overlap_rate": rate,
            }
            rows.append(item)
            _add_row(
                csv_rows,
                "ranking_overlap",
                left,
                right,
                scope,
                None,
                "overlap_rate",
                rate,
                len(left_set),
                None,
            )
    return rows


def _repeated_symbols(frame: pd.DataFrame, dates: list[str]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    target = frame.loc[frame["as_of_date"].isin(dates)].copy()
    for scope in ("top10", "top20", "top50"):
        scoped = _scope_frame(target, scope)
        grouped = scoped.groupby("symbol", dropna=False)
        rows: list[dict[str, Any]] = []
        for symbol, group in grouped:
            dates_seen = sorted(group["as_of_date"].dropna().astype(str).unique().tolist())
            rows.append({
                "symbol": str(symbol),
                "stock_name": next(
                    (str(item) for item in group["stock_name"].dropna().tolist() if str(item).strip()),
                    None,
                ),
                "appearances": len(dates_seen),
                "dates": dates_seen,
            })
        result[scope] = sorted(
            rows, key=lambda item: (-item["appearances"], item["symbol"])
        )[:20]
    return result


def _diagnostic_hints(
    summary: dict[str, Any],
    success_date: str,
    failure_date: str,
) -> list[str]:
    hints: list[str] = []
    tag_cmp = summary.get("risk_tag_analysis", {}).get("success_vs_failure_tag_ratio", {})
    top50_tags = tag_cmp.get("top50", {})
    for tag, hint in (
        ("extreme_risk_score", "failure_associated_with_extreme_risk"),
        ("strong_quant_disagree", "failure_associated_with_quant_disagreement"),
        ("quant_disagree", "failure_associated_with_quant_disagreement"),
        ("weak_market_breadth", "failure_associated_with_weak_breadth"),
        ("overheat", "failure_associated_with_overheat"),
    ):
        cmp_item = top50_tags.get(tag, {})
        s = cmp_item.get("success_ratio")
        f = cmp_item.get("failure_ratio")
        if s is not None and f is not None and f > s + 0.10 and hint not in hints:
            hints.append(hint)

    feature_cmp = summary.get("success_vs_failure", {}).get("top50", {})
    breadth = feature_cmp.get("breadth_up_ratio_5d", {})
    if (
        breadth.get("success_mean") is not None
        and breadth.get("failure_mean") is not None
        and breadth["failure_mean"] < breadth["success_mean"] - 0.05
        and "failure_associated_with_weak_breadth" not in hints
    ):
        hints.append("failure_associated_with_weak_breadth")

    overlaps = summary.get("ranking_overlap", [])
    if any(item.get("overlap_rate") is not None and item["overlap_rate"] >= 0.50 for item in overlaps):
        hints.append("high_overlap_between_dates")

    if not hints:
        hints.append("no_clear_existing_feature_separates_success_and_failure")
    return hints


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose research-only ML shadow outcomes")
    parser.add_argument("--outcomes-file", default="data/ml/shadow_mode/ml_shadow_outcomes.csv")
    parser.add_argument("--history-file", default="data/ml/shadow_mode/ml_shadow_history.csv")
    parser.add_argument("--output-dir", default="data/ml/shadow_mode")
    parser.add_argument("--success-date", default="2026-07-07", type=_date)
    parser.add_argument("--failure-date", default="2026-07-09", type=_date)
    parser.add_argument("--primary-horizon", default=5, type=int)
    parser.add_argument("--dates", nargs="*")
    args = parser.parse_args()

    outcomes_file = Path(args.outcomes_file)
    history_file = Path(args.history_file)
    if not outcomes_file.exists():
        parser.error(f"outcomes file not found: {outcomes_file}")
    if not history_file.exists():
        parser.error(f"history file not found: {history_file}")

    dates = [_date(item) for item in (args.dates or DEFAULT_DATES)]
    for date in (args.success_date, args.failure_date):
        if date not in dates:
            dates.append(date)
    dates = sorted(dict.fromkeys(dates))

    frame = _load_inputs(outcomes_file, history_file)
    frame = frame.loc[frame["as_of_date"].isin(dates)].copy()
    missing_columns = sorted(BASE_COLUMNS.difference(frame.columns))
    csv_rows: list[dict[str, Any]] = []

    date_summaries = _date_summaries(frame, dates, args.primary_horizon, csv_rows)
    success_vs_failure, missing_feature_columns = _feature_comparison(
        frame,
        args.success_date,
        args.failure_date,
        csv_rows,
    )
    risk_tag_analysis = _tag_analysis(
        frame,
        dates,
        args.success_date,
        args.failure_date,
        csv_rows,
    )
    quant_alignment = _grouped_performance(
        frame,
        dates,
        "quant_bucket",
        ["quant_lt_30", "quant_30_50", "quant_50_70", "quant_ge_70", "quant_missing"],
        csv_rows,
        "quant_alignment",
    )
    risk_level_analysis = _grouped_performance(
        frame,
        dates,
        "shadow_risk_level",
        ["low", "medium", "high", "extreme"],
        csv_rows,
        "risk_level_analysis",
    )
    board_concentration = _board_concentration(frame, dates, csv_rows)
    ranking_overlap = _ranking_overlap(frame, dates, csv_rows)
    repeated_symbols = _repeated_symbols(frame, dates)

    summary: dict[str, Any] = {
        "research_only": True,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_files": {
            "outcomes_file": str(outcomes_file),
            "history_file": str(history_file),
        },
        "success_date": args.success_date,
        "failure_date": args.failure_date,
        "primary_horizon": args.primary_horizon,
        "dates": dates,
        "date_summaries": date_summaries,
        "success_vs_failure": success_vs_failure,
        "risk_tag_analysis": risk_tag_analysis,
        "quant_alignment_analysis": quant_alignment,
        "risk_level_analysis": risk_level_analysis,
        "board_concentration": board_concentration,
        "ranking_overlap": ranking_overlap,
        "repeated_symbols": repeated_symbols,
        "missing_columns": sorted(set(missing_columns + missing_feature_columns)),
        "diagnostic_hints": [],
        "limitations": [
            "Only a small number of completed ML shadow dates are available.",
            "These diagnostics describe associations and cannot prove causality.",
            "The model's main research target is future_top30_5d.",
            "Results may still be dominated by market environment.",
            "Do not use these diagnostics to connect ML shadow signals to formal recommendations.",
        ],
    }
    summary["diagnostic_hints"] = _diagnostic_hints(
        summary,
        args.success_date,
        args.failure_date,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / "ml_shadow_diagnostics.csv"
    output_json = output_dir / "ml_shadow_diagnostics_summary.json"
    pd.DataFrame(csv_rows, columns=[
        "analysis_type",
        "as_of_date",
        "comparison_date",
        "scope",
        "group",
        "metric",
        "value",
        "sample_count",
        "completed_count",
        "note",
    ]).to_csv(output_csv, index=False, encoding="utf-8-sig")
    output_json.write_text(
        json.dumps(_clean_json(summary), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    print("StockLens ML Shadow Outcome Diagnostics")
    print(f"dates: {', '.join(dates)}")
    print(f"success_date: {args.success_date}")
    print(f"failure_date: {args.failure_date}")
    print(f"diagnostic_hints: {', '.join(summary['diagnostic_hints'])}")
    for date in (args.success_date, args.failure_date):
        top50 = date_summaries.get(date, {}).get("scopes", {}).get("top50", {})
        print(
            f"{date} top50 avg_return_{args.primary_horizon}d: "
            f"{top50.get(f'avg_return_{args.primary_horizon}d')}"
        )
    print(f"output_csv: {output_csv}")
    print(f"output_json: {output_json}")


if __name__ == "__main__":
    main()
