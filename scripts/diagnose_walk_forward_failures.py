from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.ml.ml_preprocess import MLPreprocessor


FOCUS_FEATURES = [
    "return_1d", "return_3d", "return_5d", "return_10d", "return_20d",
    "trend_score", "momentum_score", "volume_score", "risk_score",
    "overheat_score", "volatility_20d", "max_drawdown_20d", "rsi_14",
    "macd_hist", "quant_score", "market_hs300_return_5d",
    "market_zz500_return_5d", "market_cyb_return_5d", "breadth_up_ratio_5d",
    "breadth_above_ma20_ratio", "breadth_median_return_5d",
    "style_momentum_spread_5d", "style_volatility_spread_5d",
    "style_activity_spread_5d", "relative_to_hs300_return_5d",
    "relative_to_zz500_return_5d", "relative_to_breadth_median_return_5d",
]

DETAIL_COLUMNS = [
    "sample_id", "symbol", "stock_name", "as_of_date", "predicted_probability",
    "predicted_class", "probability_rank", "bucket", "future_return_5d",
    "future_excess_return_5d", "future_rank_pct_5d", "future_top30_5d",
] + FOCUS_FEATURES

DECILE_COLUMNS = [
    "decile", "count", "avg_future_return_5d", "avg_future_excess_return_5d",
    "hit_rate", "avg_probability", "avg_risk_score", "avg_overheat_score",
    "avg_return_5d", "avg_return_20d", "avg_relative_to_hs300_return_5d",
    "avg_breadth_median_return_5d",
]


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce")


def _mean(frame: pd.DataFrame, column: str) -> float | None:
    value = _numeric(frame, column).mean()
    return None if pd.isna(value) else float(value)


def _hit_rate(frame: pd.DataFrame) -> float | None:
    returns = _numeric(frame, "future_return_5d")
    valid = returns.notna()
    return None if not valid.any() else float(returns.loc[valid].gt(0).mean())


def _json_write(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False, default=str),
        encoding="utf-8",
    )


def _split_probability_buckets(valid: pd.DataFrame) -> pd.DataFrame:
    work = valid.sort_values("predicted_probability", kind="stable").copy()
    count = len(work)
    edge = max(1, int(math.floor(count * 0.20)))
    if edge * 2 >= count:
        edge = max(1, count // 3)
    work["bucket"] = "middle60"
    work.iloc[:edge, work.columns.get_loc("bucket")] = "bottom20"
    work.iloc[count - edge :, work.columns.get_loc("bucket")] = "top20"
    work["probability_rank"] = work["predicted_probability"].rank(
        ascending=False, method="first"
    ).astype(int)
    decile_count = min(10, count)
    work["decile"] = (
        pd.qcut(
            work["predicted_probability"].rank(method="first"),
            q=decile_count,
            labels=False,
        ).astype(int) + 1
    )
    return work


def _detail_frame(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [column for column in DETAIL_COLUMNS if column in frame.columns]
    return frame.loc[:, columns]


def _feature_bucket_means(work: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature in FOCUS_FEATURES:
        if feature not in work.columns:
            continue
        values = {
            bucket: _mean(work.loc[work["bucket"].eq(bucket)], feature)
            for bucket in ("top20", "middle60", "bottom20")
        }
        top, bottom = values["top20"], values["bottom20"]
        rows.append({
            "feature": feature,
            "top20_mean": top,
            "middle60_mean": values["middle60"],
            "bottom20_mean": bottom,
            "top_minus_bottom": None if top is None or bottom is None else top - bottom,
        })
    return pd.DataFrame(rows, columns=[
        "feature", "top20_mean", "middle60_mean", "bottom20_mean",
        "top_minus_bottom",
    ])


def _decile_metrics(work: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for decile, group in work.groupby("decile", sort=True):
        rows.append({
            "decile": int(decile),
            "count": int(len(group)),
            "avg_future_return_5d": _mean(group, "future_return_5d"),
            "avg_future_excess_return_5d": _mean(group, "future_excess_return_5d"),
            "hit_rate": _hit_rate(group),
            "avg_probability": _mean(group, "predicted_probability"),
            "avg_risk_score": _mean(group, "risk_score"),
            "avg_overheat_score": _mean(group, "overheat_score"),
            "avg_return_5d": _mean(group, "return_5d"),
            "avg_return_20d": _mean(group, "return_20d"),
            "avg_relative_to_hs300_return_5d": _mean(
                group, "relative_to_hs300_return_5d"
            ),
            "avg_breadth_median_return_5d": _mean(
                group, "breadth_median_return_5d"
            ),
        })
    return pd.DataFrame(rows, columns=DECILE_COLUMNS)


def _coefficients(pipeline, features: list[str]) -> tuple[pd.DataFrame, float]:
    model = pipeline.named_steps["model"]
    coefficients = model.coef_[0]
    intercept = float(model.intercept_[0])
    frame = pd.DataFrame({
        "feature": features,
        "coefficient": coefficients,
    })
    frame["abs_coefficient"] = frame["coefficient"].abs()
    frame["direction"] = np.where(frame["coefficient"].ge(0), "positive", "negative")
    frame["intercept"] = intercept
    return frame.sort_values("abs_coefficient", ascending=False, kind="stable"), intercept


def _performance(work: pd.DataFrame) -> dict[str, float | None]:
    groups = {
        bucket: work.loc[work["bucket"].eq(bucket)]
        for bucket in ("top20", "middle60", "bottom20")
    }
    result: dict[str, float | None] = {}
    for bucket, group in groups.items():
        result[f"{bucket}_avg_return"] = _mean(group, "future_return_5d")
        result[f"{bucket}_avg_excess_return"] = _mean(group, "future_excess_return_5d")
        result[f"{bucket}_hit_rate"] = _hit_rate(group)
    top = result["top20_avg_return"]
    bottom = result["bottom20_avg_return"]
    result["top_bottom_spread"] = (
        None if top is None or bottom is None else top - bottom
    )
    top_excess = result["top20_avg_excess_return"]
    bottom_excess = result["bottom20_avg_excess_return"]
    result["top_bottom_excess_spread"] = (
        None if top_excess is None or bottom_excess is None
        else top_excess - bottom_excess
    )
    return result


def _market_context(
    window: pd.Series,
    work: pd.DataFrame,
    performance: dict[str, float | None],
) -> tuple[dict[str, Any], dict[str, bool]]:
    top = work.loc[work["bucket"].eq("top20")]
    breadth = _mean(work, "breadth_up_ratio_5d")
    momentum_style = _mean(work, "style_momentum_spread_5d")
    top_return_20d = _mean(top, "return_20d")
    top_overheat = _mean(top, "overheat_score")
    top_relative = _mean(top, "relative_to_hs300_return_5d")
    top_future = performance.get("top20_avg_return")
    flags = {
        "market_weak_breadth": breadth is not None and breadth < 0.40,
        "momentum_style_failed": momentum_style is not None and momentum_style < 0,
        "possible_chasing": top_return_20d is not None and top_return_20d > 0.15,
        "possible_overheat": top_overheat is not None and top_overheat > 50,
        "relative_strength_reversal": (
            top_relative is not None and top_relative > 0.05
            and top_future is not None and top_future <= 0
        ),
    }
    hints = [name for name, active in flags.items() if active]
    if not hints:
        hints = ["no_simple_failure_hint"]
    summary = {
        "window_id": str(window["window_id"]),
        "valid_start": str(window["valid_start"]),
        "valid_end": str(window["valid_end"]),
        "status": str(window["status"]),
        "top20_avg_return": performance.get("top20_avg_return"),
        "middle60_avg_return": performance.get("middle60_avg_return"),
        "bottom20_avg_return": performance.get("bottom20_avg_return"),
        "top_bottom_spread": performance.get("top_bottom_spread"),
        "avg_market_hs300_return_5d": _mean(work, "market_hs300_return_5d"),
        "avg_market_zz500_return_5d": _mean(work, "market_zz500_return_5d"),
        "avg_market_cyb_return_5d": _mean(work, "market_cyb_return_5d"),
        "avg_breadth_up_ratio_5d": breadth,
        "avg_breadth_above_ma20_ratio": _mean(work, "breadth_above_ma20_ratio"),
        "avg_style_momentum_spread_5d": momentum_style,
        "avg_style_volatility_spread_5d": _mean(work, "style_volatility_spread_5d"),
        "interpretation_hint": hints,
    }
    return summary, flags


def _diagnose_window(
    dataset: pd.DataFrame,
    dates: pd.Series,
    window: pd.Series,
    target: str,
    model_name: str,
    output_root: Path,
) -> dict[str, Any]:
    train_end = pd.Timestamp(window["train_end"])
    valid_start = pd.Timestamp(window["valid_start"])
    valid_end = pd.Timestamp(window["valid_end"])
    target_values = pd.to_numeric(dataset[target], errors="coerce")
    train = dataset.loc[dates.le(train_end) & target_values.notna()].copy()
    valid = dataset.loc[
        dates.ge(valid_start) & dates.le(valid_end) & target_values.notna()
    ].copy()
    if train.empty or valid.empty:
        raise ValueError("window has no usable train or validation samples")

    preprocessor = MLPreprocessor()
    features = preprocessor.select_features(train, target)
    y_train = pd.to_numeric(train[target], errors="coerce").astype(int)
    if y_train.nunique() < 2:
        raise ValueError("training window contains only one target class")
    pipeline = preprocessor.build_pipeline(model_name)
    pipeline.fit(preprocessor.numeric_frame(train, features), y_train)
    probabilities = pipeline.predict_proba(
        preprocessor.numeric_frame(valid, features)
    )[:, 1]
    valid["predicted_probability"] = probabilities
    valid["predicted_class"] = (probabilities >= 0.5).astype(int)
    work = _split_probability_buckets(valid)

    top = work.loc[work["bucket"].eq("top20")].sort_values(
        "predicted_probability", ascending=False, kind="stable"
    )
    middle = work.loc[work["bucket"].eq("middle60")].sort_values(
        "predicted_probability", ascending=False, kind="stable"
    )
    bottom = work.loc[work["bucket"].eq("bottom20")].sort_values(
        "predicted_probability", ascending=True, kind="stable"
    )
    coefficients, intercept = _coefficients(pipeline, features)
    feature_means = _feature_bucket_means(work)
    deciles = _decile_metrics(work)
    performance = _performance(work)
    market_summary, flags = _market_context(window, work, performance)

    window_dir = output_root / str(window["window_id"])
    window_dir.mkdir(parents=True, exist_ok=True)
    _detail_frame(top).to_csv(window_dir / "top20_samples.csv", index=False, encoding="utf-8-sig")
    _detail_frame(bottom).to_csv(
        window_dir / "bottom20_samples.csv", index=False, encoding="utf-8-sig"
    )
    _detail_frame(middle).to_csv(
        window_dir / "middle60_samples.csv", index=False, encoding="utf-8-sig"
    )
    deciles.to_csv(window_dir / "decile_metrics.csv", index=False, encoding="utf-8-sig")
    feature_means.to_csv(
        window_dir / "feature_bucket_means.csv", index=False, encoding="utf-8-sig"
    )
    coefficients.to_csv(
        window_dir / "logistic_coefficients.csv", index=False, encoding="utf-8-sig"
    )
    _json_write(window_dir / "market_context_summary.json", market_summary)
    diagnostic_summary = {
        "window_id": str(window["window_id"]),
        "status": str(window["status"]),
        "train_end": str(window["train_end"]),
        "valid_start": str(window["valid_start"]),
        "valid_end": str(window["valid_end"]),
        "train_samples": int(len(train)),
        "valid_samples": int(len(valid)),
        "feature_count": len(features),
        "logistic_intercept": intercept,
        **performance,
        **flags,
        "interpretation_hint": market_summary["interpretation_hint"],
        "output_dir": str(window_dir),
    }
    _json_write(window_dir / "diagnostic_summary.json", diagnostic_summary)
    return diagnostic_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose failed walk-forward windows")
    parser.add_argument(
        "--dataset",
        default="data/ml/ml_research_dataset_liquid500_context_relative_2025_2026.csv",
    )
    parser.add_argument(
        "--windows-csv",
        default="data/ml/walk_forward/walk_forward_liquid500_context_top30_5d_windows.csv",
    )
    parser.add_argument("--target", default="future_top30_5d")
    parser.add_argument("--model", default="logistic", choices=["logistic"])
    parser.add_argument("--output-dir", default="data/ml/walk_forward/diagnostics")
    parser.add_argument("--windows", nargs="*")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    windows_path = Path(args.windows_csv)
    if not dataset_path.exists():
        parser.error(f"dataset not found: {dataset_path}")
    if not windows_path.exists():
        parser.error(f"windows csv not found: {windows_path}")
    dataset = pd.read_csv(dataset_path, encoding="utf-8-sig")
    windows = pd.read_csv(windows_path, encoding="utf-8-sig")
    if args.target not in dataset.columns or "as_of_date" not in dataset.columns:
        parser.error("dataset must contain target and as_of_date")
    if "window_id" not in windows.columns or "status" not in windows.columns:
        parser.error("windows csv must contain window_id and status")

    if args.windows:
        requested = set(args.windows)
        missing = sorted(requested.difference(set(windows["window_id"].astype(str))))
        if missing:
            parser.error(f"window ids not found: {missing}")
        selected = windows.loc[windows["window_id"].astype(str).isin(requested)].copy()
    else:
        selected = windows.loc[windows["status"].astype(str).eq("failed")].copy()
    if selected.empty:
        print("No matching failed walk-forward windows.")
        return

    dates = pd.to_datetime(dataset["as_of_date"], errors="coerce")
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    for _, window in selected.iterrows():
        try:
            result = _diagnose_window(
                dataset, dates, window, args.target, args.model, output_root
            )
        except Exception as exc:
            print(f"{window['window_id']}: diagnostic failed: {type(exc).__name__}: {exc}")
            continue
        print(f"window_id: {result['window_id']}")
        print(f"status: {result['status']}")
        print(f"top20_avg_return: {result['top20_avg_return']}")
        print(f"bottom20_avg_return: {result['bottom20_avg_return']}")
        print(f"top_bottom_spread: {result['top_bottom_spread']}")
        print(f"possible_chasing: {result['possible_chasing']}")
        print(f"possible_overheat: {result['possible_overheat']}")
        print(f"momentum_style_failed: {result['momentum_style_failed']}")
        print(f"output_dir: {result['output_dir']}")


if __name__ == "__main__":
    main()
