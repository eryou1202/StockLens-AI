from __future__ import annotations

import argparse

import pandas as pd

from src.ml.ml_model_registry import ResearchModelRegistry


def main() -> None:
    parser = argparse.ArgumentParser(description="列出 research model registry")
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()
    registry = ResearchModelRegistry()
    records = registry.list_models(args.limit)
    print("StockLens Research Models")
    print(f"registry: {registry.database_path}")
    print(f"count: {len(records)}")
    if not records:
        print("暂无研究模型。")
        return
    columns = [
        "model_id", "model_name", "model_type", "target", "train_end",
        "valid_start", "valid_end", "status", "created_at", "model_path",
    ]
    print(pd.DataFrame(records)[columns].to_string(index=False))


if __name__ == "__main__":
    main()
