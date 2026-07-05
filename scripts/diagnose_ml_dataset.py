from __future__ import annotations

from pathlib import Path

from src.evaluation.model_diagnostics import ModelDiagnostics


def main() -> None:
    dataset_path = Path("data/ml_dataset.csv")
    if not dataset_path.exists():
        print("data/ml_dataset.csv 不存在，请先运行：")
        print("py -m scripts.build_ml_dataset")
        return
    try:
        ModelDiagnostics(str(dataset_path)).print_report()
    except Exception as exc:
        print(f"数据集诊断失败：{type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
