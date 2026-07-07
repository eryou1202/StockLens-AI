from __future__ import annotations

import argparse
import json

from src.ml.ml_model_registry import ResearchModelRegistry


def main() -> None:
    parser = argparse.ArgumentParser(description="查看 research model 指标")
    parser.add_argument("--model-id", required=True)
    args = parser.parse_args()
    registry = ResearchModelRegistry()
    record = registry.get(args.model_id)
    if record is None:
        print(f"model not found: {args.model_id}")
        return
    print(f"model_id: {record['model_id']}")
    print(f"model_name: {record['model_name']}")
    print(f"status: {record['status']}")
    print(f"target: {record['target']}")
    print(f"model_path: {record['model_path']}")
    print("features:")
    print(json.dumps(record.get("features"), ensure_ascii=False, indent=2))
    print("metrics:")
    print(json.dumps(record.get("metrics"), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
