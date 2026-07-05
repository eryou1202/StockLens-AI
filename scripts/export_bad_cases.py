from __future__ import annotations

from pathlib import Path

from src.evaluation.case_inspector import CaseInspector


def main() -> None:
    dataset_path = Path("data/ml_dataset.csv")
    if not dataset_path.exists():
        print("data/ml_dataset.csv 不存在，请先运行：py -m scripts.build_ml_dataset")
        return

    inspector = CaseInspector(str(dataset_path))
    try:
        complete = inspector.load_complete()
    except Exception as exc:
        print(f"案例数据读取失败：{type(exc).__name__}: {exc}")
        return
    if complete.empty:
        print("complete 样本为空，暂时没有可导出的错误案例。")
        return

    output_dir = Path("data/diagnostics")
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "bad_support_cases.csv": inspector.top_bad_support_cases(),
        "missed_reject_cases.csv": inspector.top_missed_reject_cases(),
        "extreme_score_mismatch_cases.csv": inspector.extreme_score_mismatch_cases(),
    }
    print("StockLens Bad Case Export\n")
    for filename, frame in outputs.items():
        path = output_dir / filename
        frame.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"{path}: {len(frame)} rows")
    print("\n当前历史样本来自测试型 historical batch，不代表真实历史回测表现。")


if __name__ == "__main__":
    main()
