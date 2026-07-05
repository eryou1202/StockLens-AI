from __future__ import annotations

import argparse
from datetime import datetime

from src.config.settings import load_settings
from src.portfolio.position_manager import PositionManager
from src.portfolio.position_schema import Position


def _date(value: str) -> datetime:
    try:
        return datetime.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("日期格式应为 YYYY-MM-DD 或 ISO datetime") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description="Add a local StockLens position.")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--name")
    parser.add_argument("--entry-price", required=True, type=float)
    parser.add_argument("--entry-date", required=True, type=_date)
    parser.add_argument("--size", type=float)
    parser.add_argument("--reason")
    parser.add_argument("--stop-loss", type=float)
    parser.add_argument("--take-profit", type=float)
    parser.add_argument("--max-holding-days", type=int)
    args = parser.parse_args()

    settings = load_settings()
    manager = PositionManager(settings.database_path)
    try:
        position_id = manager.add_position(
            Position(
                symbol=args.symbol,
                stock_name=args.name,
                entry_date=args.entry_date,
                entry_price=args.entry_price,
                position_size=args.size,
                entry_reason=args.reason,
                entry_action="manual",
                stop_loss_price=args.stop_loss,
                take_profit_price=args.take_profit,
                max_holding_days=args.max_holding_days,
            )
        )
    except Exception as exc:
        print(f"新增持仓失败：{exc}")
        return
    print(f"持仓已添加：id={position_id}, symbol={args.symbol.upper()}")


if __name__ == "__main__":
    main()
