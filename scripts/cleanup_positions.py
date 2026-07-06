from __future__ import annotations

from src.config.settings import load_settings
from src.portfolio.position_manager import PositionManager


def main() -> None:
    count = PositionManager(load_settings().database_path).cleanup_test_positions()
    print(f"已将 entry_price <= 0.01 且 status=open 的异常测试持仓转换为 watch_only：{count} 条。")


if __name__ == "__main__":
    main()
