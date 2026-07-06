from __future__ import annotations

import argparse
from src.config.settings import load_settings
from src.tracking.recommendation_tracker import RecommendationTracker
from src.tracking.tracking_schema import ManualVerdict


def main() -> None:
    parser = argparse.ArgumentParser(description="Mark a human verdict for one recommendation snapshot.")
    parser.add_argument("--id", required=True, type=int)
    parser.add_argument("--verdict", required=True, choices=[item.value for item in ManualVerdict])
    parser.add_argument("--notes")
    args = parser.parse_args()
    try:
        RecommendationTracker(load_settings().database_path).mark_verdict(args.id, args.verdict, args.notes)
        print(f"追踪记录 id={args.id} 已标记 verdict={args.verdict}。")
    except Exception as exc:
        print(f"标记失败：{type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
