from __future__ import annotations

import argparse
from src.recommendation.candidate_pool import CandidatePoolEditor


def main() -> None:
    parser = argparse.ArgumentParser(description="Add a manual watch candidate to Signal Package v1.0.")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--name")
    parser.add_argument("--notes")
    parser.add_argument("--horizons", nargs="+", type=int, default=[3, 5])
    args = parser.parse_args()
    try:
        CandidatePoolEditor().add_manual_watch(args.symbol, args.name, args.notes, args.horizons)
        print(f"人工观察股已添加：{args.symbol.upper()}，source_type=manual_watch")
    except Exception as exc:
        print(f"添加失败：{type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
