from __future__ import annotations

from src.config.settings import load_settings
from src.dataset.ml_dataset_builder import MLDatasetBuilder


def main() -> None:
    settings = load_settings()
    output_path = "data/ml_dataset.csv"
    print("StockLens ML Dataset Builder\n")
    print(f"database: {settings.database_path}")
    print(f"output: {output_path}\n")

    try:
        frame = MLDatasetBuilder(settings.database_path).build(output_path)
    except Exception as exc:
        print(f"数据集生成失败：{type(exc).__name__}: {exc}")
        return

    status = frame["feedback_status"] if "feedback_status" in frame else []
    print(f"rows: {len(frame)}")
    print(f"columns: {len(frame.columns)}")
    print(f"complete labels: {int((status == 'complete').sum()) if len(frame) else 0}")
    print(f"partial labels: {int((status == 'partial').sum()) if len(frame) else 0}")
    print(f"pending labels: {int((status == 'pending').sum()) if len(frame) else 0}")
    print(f"failed labels: {int((status == 'failed').sum()) if len(frame) else 0}")


if __name__ == "__main__":
    main()
