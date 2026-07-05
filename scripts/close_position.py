from __future__ import annotations

import argparse
from datetime import datetime

from src.config.settings import load_settings
from src.portfolio.position_manager import PositionManager


def _date(value: str) -> datetime:
    try:
        return datetime.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("日期格式应为 YYYY-MM-DD 或 ISO datetime") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description="Close a local StockLens position.")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--exit-price", required=True, type=float)
    parser.add_argument("--exit-date", required=True, type=_date)
    parser.add_argument("--reason")
    args = parser.parse_args()
    try:
        PositionManager(load_settings().database_path).close_position(
            symbol=args.symbol,
            exit_price=args.exit_price,
            exit_date=args.exit_date,
            exit_reason=args.reason,
        )
    except Exception as exc:
        print(f"关闭持仓失败：{exc}")
        return
    print(f"持仓已关闭：{args.symbol.upper()}, exit_price={args.exit_price:.4f}")


if __name__ == "__main__":
    main()
