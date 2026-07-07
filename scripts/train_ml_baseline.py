from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from src.ml.ml_schema import MLTrainRequest
from src.ml.ml_trainer import MLTrainer


def _date(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d")


def _print_metrics(metrics: dict) -> None:
    print("validation metrics:")
    for key, value in metrics.items():
        if key in {"buckets", "mean_future_return_by_pred_bucket"}:
            continue
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")
    if metrics.get("buckets"):
        print("ranking buckets:")
        for bucket in metrics["buckets"]:
            print(
                f"  {bucket['bucket']}: count={bucket['count']}, "
                f"avg_return={bucket['avg_future_return']}, "
                f"hit_rate={bucket['hit_rate']}, avg_drawdown={bucket['avg_max_drawdown']}"
            )


def _legacy_main() -> None:
    """Keep the pre-v1.0 no-argument diagnostics command working."""
    from src.ml.ml_baseline import MLBaselineTrainer

    dataset = Path("data/ml_dataset.csv")
    print("StockLens legacy ML Baseline Diagnostics")
    print(
        "Legacy ML baseline does not belong to ML Research Foundation v1.0 "
        "and does not use ResearchModelRegistry."
    )
    if not dataset.exists():
        print("data/ml_dataset.csv 不存在，请先运行 python -m scripts.build_ml_dataset")
        return
    result = MLBaselineTrainer(str(dataset)).train(target="hit_5d")
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    print("当前结果来自测试型历史样本，不代表真实历史回测表现。")


def main() -> None:
    if len(sys.argv) == 1:
        _legacy_main()
        return
    parser = argparse.ArgumentParser(description="训练隔离的 ML research baseline")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument(
        "--model", required=True, choices=["logistic", "random_forest_regressor"]
    )
    parser.add_argument("--train-end", required=True, type=_date)
    parser.add_argument("--valid-start", required=True, type=_date)
    parser.add_argument("--valid-end", required=True, type=_date)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--notes")
    args = parser.parse_args()

    request = MLTrainRequest(
        dataset_path=args.dataset,
        target=args.target,
        model_type=args.model,
        train_end=args.train_end,
        valid_start=args.valid_start,
        valid_end=args.valid_end,
        model_name=args.model_name,
        notes=args.notes,
    )
    try:
        result = MLTrainer().train(request)
    except Exception as exc:
        print(f"training failed: {type(exc).__name__}: {exc}")
        return
    print("StockLens ML Research Baseline")
    print(f"status: {result.get('status')}")
    if result.get("message"):
        print(result["message"])
        return
    print(f"model_id: {result['model_id']}")
    print(f"research_only: {result['research_only']}")
    print(f"train_samples: {result['train_samples']}")
    print(f"valid_samples: {result['valid_samples']}")
    print(f"feature_count: {result['feature_count']}")
    _print_metrics(result["metrics"])
    print(f"model_path: {result['model_path']}")
    print(f"registry_path: {result['registry_path']}")
    print("该模型仅处于 research 状态，未接入正式推荐。")


if __name__ == "__main__":
    main()
