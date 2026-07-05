from __future__ import annotations

import argparse

from rich.console import Console
from rich.table import Table

from src.config.settings import load_settings
from src.portfolio.position_manager import PositionManager


def main() -> None:
    parser = argparse.ArgumentParser(description="List local StockLens positions.")
    parser.add_argument("--status", choices=["open", "closed", "watch_only", "all"], default="open")
    args = parser.parse_args()
    positions = PositionManager(load_settings().database_path).list_positions(args.status)
    if not positions:
        print(f"没有 status={args.status} 的持仓。")
        return

    table = Table(title=f"StockLens Positions ({args.status})")
    for column in ("id", "symbol", "name", "status", "entry_date", "entry_price", "size", "stop_loss", "take_profit", "exit_price"):
        table.add_column(column)
    for item in positions:
        table.add_row(
            str(item.id), item.symbol, item.stock_name or "-", item.status.value,
            item.entry_date.isoformat(), f"{item.entry_price:.4f}",
            "-" if item.position_size is None else f"{item.position_size:g}",
            "-" if item.stop_loss_price is None else f"{item.stop_loss_price:.4f}",
            "-" if item.take_profit_price is None else f"{item.take_profit_price:.4f}",
            "-" if item.exit_price is None else f"{item.exit_price:.4f}",
        )
    Console().print(table)


if __name__ == "__main__":
    main()
