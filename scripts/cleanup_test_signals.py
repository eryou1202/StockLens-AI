from __future__ import annotations

import argparse
import sqlite3
from datetime import date
from pathlib import Path
from typing import Any

from src.config.settings import load_settings


def _parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("日期格式应为 YYYY-MM-DD") from exc


def _build_conditions(args: argparse.Namespace) -> tuple[list[str], list[Any], list[str], list[Any]]:
    snapshot_conditions: list[str] = []
    snapshot_params: list[Any] = []
    feedback_conditions: list[str] = []
    feedback_params: list[Any] = []

    if args.before:
        limit = f"{args.before.isoformat()}T00:00:00"
        snapshot_conditions.append("as_of_time < ?")
        snapshot_params.append(limit)
        feedback_conditions.append("as_of_time < ?")
        feedback_params.append(limit)
    if args.after:
        limit = f"{args.after.isoformat()}T23:59:59.999999"
        snapshot_conditions.append("as_of_time > ?")
        snapshot_params.append(limit)
        feedback_conditions.append("as_of_time > ?")
        feedback_params.append(limit)
    if args.status:
        snapshot_conditions.append(
            "EXISTS ("
            "SELECT 1 FROM signal_feedback AS f "
            "WHERE f.symbol = signal_snapshots.symbol "
            "AND f.as_of_time = signal_snapshots.as_of_time "
            "AND f.feedback_status = ?"
            ")"
        )
        snapshot_params.append(args.status)
        feedback_conditions.append("feedback_status = ?")
        feedback_params.append(args.status)

    return snapshot_conditions, snapshot_params, feedback_conditions, feedback_params


def main() -> None:
    parser = argparse.ArgumentParser(description="Safely preview or remove StockLens test signals.")
    parser.add_argument("--before", type=_parse_date)
    parser.add_argument("--after", type=_parse_date)
    parser.add_argument("--status", choices=["pending", "partial", "complete", "failed"])
    parser.add_argument("--dry-run", action="store_true", help="只预览，不删除（默认行为）")
    parser.add_argument("--confirm", action="store_true", help="确认执行删除")
    args = parser.parse_args()

    if not any((args.before, args.after, args.status)):
        print("拒绝执行：必须至少提供 --before、--after 或 --status 之一。")
        return
    if args.before and args.after and args.after >= args.before:
        print("拒绝执行：同时指定时，--after 必须早于 --before。")
        return

    settings = load_settings()
    database_path = Path(settings.database_path)
    if not database_path.exists():
        print(f"数据库不存在：{database_path}")
        return

    snapshot_conditions, snapshot_params, feedback_conditions, feedback_params = _build_conditions(args)
    snapshot_where = " AND ".join(snapshot_conditions)
    feedback_where = " AND ".join(feedback_conditions)

    try:
        connection = sqlite3.connect(database_path)
        tables = {
            row[0]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        if "signal_snapshots" not in tables or "signal_feedback" not in tables:
            print("数据库缺少 signal_snapshots 或 signal_feedback 表，无法执行清理。")
            return

        snapshot_count = connection.execute(
            f"SELECT COUNT(*) FROM signal_snapshots WHERE {snapshot_where}",
            snapshot_params,
        ).fetchone()[0]
        feedback_count = connection.execute(
            f"SELECT COUNT(*) FROM signal_feedback WHERE {feedback_where}",
            feedback_params,
        ).fetchone()[0]

        print("StockLens Test Signal Cleanup\n")
        print(f"database: {database_path}")
        print(f"matching signal_snapshots: {snapshot_count}")
        print(f"matching signal_feedback: {feedback_count}")

        if not args.confirm or args.dry_run:
            print("mode: dry-run")
            print("未删除任何数据；如确认无误，请在相同条件后添加 --confirm。")
            return

        with connection:
            # status 条件依赖反馈表，必须先删快照再删反馈。
            deleted_snapshots = connection.execute(
                f"DELETE FROM signal_snapshots WHERE {snapshot_where}",
                snapshot_params,
            ).rowcount
            deleted_feedback = connection.execute(
                f"DELETE FROM signal_feedback WHERE {feedback_where}",
                feedback_params,
            ).rowcount
        print("mode: confirmed delete")
        print(f"deleted signal_snapshots: {deleted_snapshots}")
        print(f"deleted signal_feedback: {deleted_feedback}")
    except sqlite3.Error as exc:
        print(f"清理未完成：{exc}")
    finally:
        if "connection" in locals():
            connection.close()


if __name__ == "__main__":
    main()
