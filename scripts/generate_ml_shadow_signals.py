from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.ml.ml_preprocess import MLPreprocessor


OUTPUT_COLUMNS = [
    "as_of_date", "symbol", "stock_name", "ml_model_id", "ml_model_name",
    "ml_target", "ml_score", "ml_rank", "ml_bucket", "quant_score",
    "candidate_bucket", "risk_flags", "return_5d", "return_20d",
    "volume_ratio_5d", "risk_score", "overheat_score", "breadth_up_ratio_5d",
    "breadth_above_ma20_ratio", "shadow_risk_level", "shadow_risk_tags",
    "shadow_interpretation", "shadow_action", "shadow_note",
]

SHADOW_NOTE = "Research-only shadow signal; not a formal recommendation or investment advice."


def _date(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d")


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if np.isfinite(result) else None


def _text(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def _risk_flags(value: Any) -> str:
    if isinstance(value, list):
        return ";".join(str(item) for item in value if str(item).strip())
    if value is None or (not isinstance(value, (list, dict)) and pd.isna(value)):
        return ""
    return str(value).strip()


def _bucket(rank: int, thresholds: list[int]) -> tuple[str, str]:
    if rank <= thresholds[0]:
        return "top10", "ml_top10_watch"
    if rank <= thresholds[1]:
        return "top20", "ml_top20_watch"
    if rank <= thresholds[2]:
        return "top50", "ml_top50_watch"
    return "outside_top50", "ml_ignore"


def _risk_annotation(row: Any) -> tuple[str, str, str]:
    tags: list[str] = []
    severity = 0

    def add(tag: str, minimum: int = 0) -> None:
        nonlocal severity
        if tag not in tags:
            tags.append(tag)
        severity = max(severity, minimum)

    risk_score = _number(row.get("risk_score"))
    return_5d = _number(row.get("return_5d"))
    return_20d = _number(row.get("return_20d"))
    quant_score = _number(row.get("quant_score"))
    breadth = _number(row.get("breadth_up_ratio_5d"))
    ma_breadth = _number(row.get("breadth_above_ma20_ratio"))
    overheat = _number(row.get("overheat_score"))
    bucket = str(row.get("ml_bucket") or "")

    if risk_score is not None and risk_score >= 70:
        add("high_risk_score", 2)
    if risk_score is not None and risk_score >= 85:
        add("extreme_risk_score", 3)
    if return_5d is not None and return_5d <= -0.10:
        add("recent_5d_drop", 2)
    if return_5d is not None and return_5d <= -0.20:
        add("extreme_5d_drop", 3)
    if (
        return_20d is not None and return_20d >= 0.30
        and return_5d is not None and return_5d <= -0.05
    ):
        add("high_volatility_reversal")
    if quant_score is not None and quant_score < 40:
        add("quant_disagree", 2)
    if quant_score is not None and quant_score < 30:
        add("strong_quant_disagree", 3)
    if breadth is not None and breadth < 0.40:
        add("weak_market_breadth", 1)
    if ma_breadth is not None and ma_breadth < 0.40:
        add("weak_ma_breadth", 1)
    if overheat is not None and overheat >= 50:
        add("overheat")
    if return_5d is not None and return_5d < 0 and bucket in {"top10", "top20"}:
        add("rebound_candidate")

    levels = ("low", "medium", "high", "extreme")
    level = levels[severity]
    if level == "extreme":
        interpretation = "极高风险，仅作 ML 影子观察，不可作为买入依据。"
    elif level == "high" and "rebound_candidate" in tags:
        interpretation = "高风险反弹观察，需等待实时确认和规则系统过滤。"
    elif level == "high":
        interpretation = "高风险 ML 观察信号，不参与正式推荐。"
    elif level == "medium":
        interpretation = "中等风险 ML 观察信号，仅供研究跟踪。"
    else:
        interpretation = "低风险 ML 观察信号，但仍不构成正式推荐。"
    return level, ";".join(tags), interpretation


def _annotate_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    annotations = [_risk_annotation(row) for _, row in result.iterrows()]
    result["shadow_risk_level"] = [item[0] for item in annotations]
    result["shadow_risk_tags"] = [item[1] for item in annotations]
    result["shadow_interpretation"] = [item[2] for item in annotations]
    return result.reindex(columns=OUTPUT_COLUMNS)


def _train_shadow_model(
    frame: pd.DataFrame,
    dates: pd.Series,
    target: str,
    train_end: pd.Timestamp,
) -> tuple[Any, list[str], int]:
    target_values = pd.to_numeric(frame[target], errors="coerce")
    train = frame.loc[dates.le(train_end) & target_values.notna()].copy()
    if train.empty:
        raise ValueError("training split is empty")
    preprocessor = MLPreprocessor()
    features = preprocessor.select_features(train, target)
    y_train = pd.to_numeric(train[target], errors="coerce").astype(int)
    if not features:
        raise ValueError("no safe numeric features")
    if y_train.nunique() < 2:
        raise ValueError("training split contains only one target class")
    pipeline = preprocessor.build_pipeline("logistic")
    pipeline.fit(preprocessor.numeric_frame(train, features), y_train)
    return pipeline, features, len(train)


def _load_model_package(
    model_path: Path,
    target: str,
) -> tuple[Any, list[str]]:
    package = joblib.load(model_path)
    if not isinstance(package, dict) or "pipeline" not in package or "features" not in package:
        raise ValueError("model package must contain pipeline and features")
    package_target = package.get("target")
    if package_target and str(package_target) != target:
        raise ValueError(f"model target mismatch: {package_target} != {target}")
    return package["pipeline"], list(package["features"])


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(frame.to_json(orient="records", force_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate research-only ML shadow signals")
    parser.add_argument(
        "--dataset",
        default="data/ml/ml_research_dataset_liquid500_context_relative_daily_2025_2026.csv",
    )
    parser.add_argument("--as-of-date", type=_date)
    parser.add_argument("--target", default="future_top30_5d")
    parser.add_argument("--model", default="logistic", choices=["logistic"])
    parser.add_argument("--train-end", type=_date)
    parser.add_argument("--top-k", nargs="+", type=int, default=[10, 20, 50])
    parser.add_argument("--model-path")
    parser.add_argument("--model-id")
    parser.add_argument("--output-dir", default="data/ml/shadow_mode")
    args = parser.parse_args()

    thresholds = sorted(set(args.top_k))
    if len(thresholds) != 3 or thresholds[0] <= 0:
        parser.error("--top-k must provide exactly three distinct positive thresholds")
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        parser.error(f"dataset not found: {dataset_path}")
    frame = pd.read_csv(dataset_path, encoding="utf-8-sig")
    required = {"as_of_date", "symbol", args.target}
    missing = sorted(required.difference(frame.columns))
    if missing:
        parser.error(f"dataset missing columns: {missing}")
    dates = pd.to_datetime(frame["as_of_date"], errors="coerce")
    available_dates = dates.dropna()
    if available_dates.empty:
        parser.error("dataset has no valid as_of_date")
    as_of = (
        pd.Timestamp(args.as_of_date)
        if args.as_of_date is not None else available_dates.max().normalize()
    )
    predict = frame.loc[dates.eq(as_of)].copy()
    if predict.empty:
        parser.error(f"dataset has no samples for {as_of.date().isoformat()}")
    earlier_dates = available_dates.loc[available_dates.lt(as_of)]
    if earlier_dates.empty and not args.model_path:
        parser.error("dataset has no date before as-of-date for training")
    train_end = (
        pd.Timestamp(args.train_end)
        if args.train_end is not None else earlier_dates.max().normalize()
    )
    if train_end >= as_of:
        parser.error("train-end must be earlier than as-of-date")

    model_name = f"shadow_{args.model}_{args.target}"
    if args.model_path:
        model_path = Path(args.model_path)
        if not model_path.exists():
            parser.error(f"model path not found: {model_path}")
        pipeline, features = _load_model_package(model_path, args.target)
        train_samples = 0
        model_id = args.model_id or model_path.stem
        model_name = f"shadow_{model_path.stem}"
    else:
        try:
            pipeline, features, train_samples = _train_shadow_model(
                frame, dates, args.target, train_end
            )
        except Exception as exc:
            parser.error(f"shadow model training failed: {type(exc).__name__}: {exc}")
        model_id = args.model_id or (
            f"shadow_{args.model}_{args.target}_{train_end.strftime('%Y%m%d')}"
        )
    try:
        predict["ml_score"] = pipeline.predict_proba(
            MLPreprocessor.numeric_frame(predict, features)
        )[:, 1]
    except Exception as exc:
        parser.error(f"shadow prediction failed: {type(exc).__name__}: {exc}")
    predict = predict.sort_values(
        ["ml_score", "symbol"], ascending=[False, True], kind="stable"
    ).reset_index(drop=True)

    rows: list[dict[str, Any]] = []
    for index, item in predict.iterrows():
        rank = index + 1
        bucket, action = _bucket(rank, thresholds)
        rows.append({
            "as_of_date": as_of.date().isoformat(),
            "symbol": str(item.get("symbol")),
            "stock_name": _text(item.get("stock_name")),
            "ml_model_id": model_id,
            "ml_model_name": model_name,
            "ml_target": args.target,
            "ml_score": _number(item.get("ml_score")),
            "ml_rank": rank,
            "ml_bucket": bucket,
            "quant_score": _number(item.get("quant_score")),
            "candidate_bucket": _text(item.get("candidate_bucket")),
            "risk_flags": _risk_flags(item.get("risk_flags")),
            "return_5d": _number(item.get("return_5d")),
            "return_20d": _number(item.get("return_20d")),
            "volume_ratio_5d": _number(item.get("volume_ratio_5d")),
            "risk_score": _number(item.get("risk_score")),
            "overheat_score": _number(item.get("overheat_score")),
            "breadth_up_ratio_5d": _number(item.get("breadth_up_ratio_5d")),
            "breadth_above_ma20_ratio": _number(item.get("breadth_above_ma20_ratio")),
            "shadow_action": action,
            "shadow_note": SHADOW_NOTE,
        })
    output = _annotate_frame(pd.DataFrame(rows))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    latest_csv = output_dir / "ml_shadow_latest.csv"
    latest_json = output_dir / "ml_shadow_latest.json"
    history_csv = output_dir / "ml_shadow_history.csv"
    output.to_csv(latest_csv, index=False, encoding="utf-8-sig")
    latest_payload = {
        "research_only": True,
        "as_of_date": as_of.date().isoformat(),
        "train_end": train_end.date().isoformat(),
        "train_samples": train_samples,
        "predict_samples": len(output),
        "top_k_thresholds": thresholds,
        "records": _records(output),
    }
    latest_json.write_text(
        json.dumps(latest_payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    if history_csv.exists():
        history = pd.read_csv(history_csv, encoding="utf-8-sig")
        history = history.reindex(columns=OUTPUT_COLUMNS)
        history = _annotate_frame(history)
    else:
        history = pd.DataFrame(columns=OUTPUT_COLUMNS)
    combined = pd.concat([history, output], ignore_index=True)
    combined = combined.drop_duplicates(
        subset=["as_of_date", "symbol", "ml_model_name"], keep="last"
    ).sort_values(["as_of_date", "ml_model_name", "ml_rank"], kind="stable")
    combined.to_csv(history_csv, index=False, encoding="utf-8-sig")

    risk_level_counts = {
        level: int(output["shadow_risk_level"].eq(level).sum())
        for level in ("low", "medium", "high", "extreme")
    }
    extreme = output["shadow_risk_level"].eq("extreme")
    top10_extreme_count = int(
        (extreme & output["ml_rank"].le(thresholds[0])).sum()
    )
    top20_extreme_count = int(
        (extreme & output["ml_rank"].le(thresholds[1])).sum()
    )
    top50_extreme_count = int(
        (extreme & output["ml_rank"].le(thresholds[2])).sum()
    )

    print("StockLens ML Shadow Signals")
    print(f"as_of_date: {as_of.date().isoformat()}")
    print(f"train_samples: {train_samples}")
    print(f"predict_samples: {len(output)}")
    print(f"top10: {min(thresholds[0], len(output))}")
    print(f"top20: {min(thresholds[1], len(output))}")
    print(f"top50: {min(thresholds[2], len(output))}")
    print(f"risk_level_counts: {risk_level_counts}")
    print(f"top10_extreme_count: {top10_extreme_count}")
    print(f"top20_extreme_count: {top20_extreme_count}")
    print(f"top50_extreme_count: {top50_extreme_count}")
    print(f"output_csv: {latest_csv}")
    print(f"output_json: {latest_json}")
    print(f"history_csv: {history_csv}")


if __name__ == "__main__":
    main()
