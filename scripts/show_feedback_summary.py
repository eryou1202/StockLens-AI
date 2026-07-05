from __future__ import annotations

from pathlib import Path

from src.evaluation.performance_evaluator import PerformanceEvaluator


def main() -> None:
    dataset_path = Path("data/ml_dataset.csv")
    if not dataset_path.exists():
        print("data/ml_dataset.csv 不存在，请先运行：")
        print("py -m scripts.build_ml_dataset")
        return

    try:
        PerformanceEvaluator(str(dataset_path)).print_summary()
    except Exception as exc:
        print(f"反馈统计读取失败：{type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
