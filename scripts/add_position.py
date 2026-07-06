from __future__ import annotations

import argparse
from datetime import datetime

from src.config.settings import load_settings
from src.portfolio.position_manager import PositionManager
from src.portfolio.position_schema import Position, PositionStatus


def _date(value: str) -> datetime:
    try:
        return datetime.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("日期格式应为 YYYY-MM-DD 或 ISO datetime") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description="Add a real position or a watch-only symbol.")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--name")
    parser.add_argument("--entry-price", type=float, default=0.01,
                        help="Omit or use <=0.01 to create watch_only.")
    parser.add_argument("--entry-date", type=_date, default=datetime.now())
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--size", type=float)
    parser.add_argument("--reason")
    parser.add_argument("--stop-loss", type=float)
    parser.add_argument("--take-profit", type=float)
    parser.add_argument("--max-holding-days", type=int)
    args = parser.parse_args()
    watch = args.watch or args.entry_price <= 0.01
    try:
        position_id = PositionManager(load_settings().database_path).add_position(Position(
            symbol=args.symbol, stock_name=args.name, entry_date=args.entry_date,
            entry_price=0.01 if watch else args.entry_price,
            position_size=None if watch else args.size, entry_reason=args.reason,
            entry_action="manual_watch" if watch else "manual",
            stop_loss_price=None if watch else args.stop_loss,
            take_profit_price=None if watch else args.take_profit,
            max_holding_days=args.max_holding_days,
            status=PositionStatus.WATCH_ONLY if watch else PositionStatus.OPEN,
            metadata={"source_type": "manual_watch"} if watch else {},
        ))
    except Exception as exc:
        print(f"新增失败：{exc}")
        return
    print(f"已添加：id={position_id}, symbol={args.symbol.upper()}, status={'watch_only' if watch else 'open'}")


if __name__ == "__main__":
    main()
