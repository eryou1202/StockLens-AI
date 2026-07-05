from __future__ import annotations

from pathlib import Path
from typing import Any

from src.ml.ml_baseline import MLBaselineTrainer


def _print_distribution(name: str, value: dict[str, int] | None) -> None:
    if value is not None:
        print(f"{name}: 0={value.get('0', 0)}, 1={value.get('1', 0)}")


def _format_optional(value: Any) -> str:
    return "-" if value is None else f"{float(value):.4f}"


def main() -> None:
    dataset_path = Path("data/ml_dataset.csv")
    print("StockLens ML Baseline Trainer\n")
    if not dataset_path.exists():
        print("data/ml_dataset.csv 不存在，请先运行：")
        print("py -m scripts.build_ml_dataset")
        return

    try:
        result = MLBaselineTrainer(str(dataset_path)).train(target="hit_5d")
    except Exception as exc:
        print(f"训练流程未完成：{type(exc).__name__}: {exc}")
        return

    status = result.get("status")
    print(f"status: {status}")
    if result.get("message"):
        print(result["message"])
    print(f"current complete samples: {result.get('complete_samples', 0)}")
    _print_distribution("label_distribution_total", result.get("label_distribution_total"))
    _print_distribution("label_distribution_train", result.get("label_distribution_train"))
    _print_distribution("label_distribution_test", result.get("label_distribution_test"))

    if status == "trained":
        _print_distribution("prediction_distribution_test", result.get("prediction_distribution_test"))
        matrix = result["confusion_matrix"]
        print(
            "confusion_matrix: "
            f"tn={matrix['tn']}, fp={matrix['fp']}, fn={matrix['fn']}, tp={matrix['tp']}"
        )
        print(f"positive_rate_test: {_format_optional(result.get('positive_rate_test'))}")
        print(f"predicted_positive_rate: {_format_optional(result.get('predicted_positive_rate'))}")
        print(f"baseline_accuracy: {_format_optional(result.get('baseline_accuracy'))}")
        print(f"model accuracy: {result['accuracy']:.4f}")
        accuracy_diff = result["model_vs_baseline"]["accuracy_diff"]
        print(f"accuracy_diff: {_format_optional(accuracy_diff)}")
        if accuracy_diff > 0:
            print("模型准确率超过测试集多数类 baseline。")
        elif accuracy_diff == 0:
            print("模型准确率仅与测试集多数类 baseline 持平。")
        else:
            print("模型准确率低于测试集多数类 baseline，当前模型没有显示增益。")
        print(f"precision: {result['precision']:.4f}")
        print(f"recall: {result['recall']:.4f}")
        print(f"roc_auc: {_format_optional(result.get('roc_auc'))}")
        print(f"intercept: {result['intercept']:.6f}")
        print("top coefficients:")
        for feature, coefficient in result.get("coefficients", {}).items():
            print(f"  {feature}: {coefficient:.6f}")
        print(f"model_path: {result['model_path']}")
    elif result.get("baseline_accuracy") is not None:
        print(f"preview baseline_accuracy: {result['baseline_accuracy']:.4f}")

    print(f"feature_columns: {', '.join(result.get('feature_columns', [])) or '-'}")
    print("\n当前结果来自测试型历史样本，不代表真实历史回测表现。")


if __name__ == "__main__":
    main()
