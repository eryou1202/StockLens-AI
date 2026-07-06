from __future__ import annotations

import argparse
from src.recommendation.candidate_pool import CandidatePoolEditor


def main() -> None:
    parser = argparse.ArgumentParser(description="Remove one candidate without touching historical data.")
    parser.add_argument("--symbol", required=True)
    args = parser.parse_args()
    try:
        count = CandidatePoolEditor().remove(args.symbol)
        print(f"已从 ai_candidates.json 删除 {count} 条：{args.symbol.upper()}。历史信号、持仓和追踪记录未修改。")
    except Exception as exc:
        print(f"删除失败：{type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
