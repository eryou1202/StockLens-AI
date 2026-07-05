from __future__ import annotations

from pathlib import Path

from src.evaluation.case_inspector import CaseInspector


def main() -> None:
    dataset_path = Path("data/ml_dataset.csv")
    if not dataset_path.exists():
        print("data/ml_dataset.csv 不存在，请先运行：py -m scripts.build_ml_dataset")
        return
    try:
        CaseInspector(str(dataset_path)).print_report(n=10)
    except Exception as exc:
        print(f"案例检查失败：{type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
