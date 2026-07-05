from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from src.config.settings import load_settings


def _format_number(value: Any) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return str(value)


def _json_object(value: Any) -> dict[str, Any]:
    try:
        parsed = json.loads(value or "{}")
    except (TypeError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _quant_decision(final_decision_json: Any) -> str:
    payload = _json_object(final_decision_json)
    quant = payload.get("quant_result") or {}
    extra = ((quant.get("features") or {}).get("extra") or {})
    score_breakdown = extra.get("score_breakdown") or {}
    value = extra.get("internal_quant_decision") or score_breakdown.get("quant_decision")
    if value:
        return str(value)
    value = quant.get("quant_decision")
    if value in {"strong_watch", "watch", "support"}:
        return "support"
    if value in {"risky", "uncertain", "neutral"}:
        return "uncertain"
    if value in {"avoid", "reject"}:
        return "reject"
    return str(payload.get("quant_view") or "-")


def main() -> None:
    settings = load_settings()
    database_path = Path(settings.database_path)
    print("StockLens Feedback Rows\n")
    print(f"database: {database_path}\n")

    if not database_path.exists():
        print("数据库不存在，请先运行：py -m scripts.run_file_candidates")
        return

    try:
        connection = sqlite3.connect(database_path)
        exists = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='signal_feedback'"
        ).fetchone()
        if not exists:
            print("signal_feedback 表不存在，请先运行：py -m scripts.update_feedback")
            return
        rows = connection.execute(
            """
            SELECT f.symbol, f.as_of_time, f.entry_trade_date, f.entry_close,
                   f.future_return_1d, f.future_return_3d,
                   f.future_return_5d, f.future_return_10d,
                   f.future_max_drawdown_5d, f.future_max_drawdown_10d,
                   f.feedback_status, f.metadata_json,
                   (
                       SELECT s.final_level FROM signal_snapshots AS s
                       WHERE s.symbol = f.symbol AND s.as_of_time = f.as_of_time
                       ORDER BY s.id DESC LIMIT 1
                   ) AS final_level,
                   (
                       SELECT s.final_decision_json FROM signal_snapshots AS s
                       WHERE s.symbol = f.symbol AND s.as_of_time = f.as_of_time
                       ORDER BY s.id DESC LIMIT 1
                   ) AS final_decision_json
            FROM signal_feedback AS f
            ORDER BY f.id DESC
            LIMIT 20
            """
        ).fetchall()
    except sqlite3.Error as exc:
        print(f"读取反馈失败：{exc}")
        return
    finally:
        if "connection" in locals():
            connection.close()

    if not rows:
        print("signal_feedback 暂无数据。")
        return

    print("最近 20 条 signal_feedback")
    for index, row in enumerate(rows, start=1):
        metadata = _json_object(row[11])
        print(f"\n[{index}]")
        print(f"symbol: {row[0]}")
        print(f"as_of_time: {row[1]}")
        print(f"final_level: {row[12] or '-'}")
        print(f"quant_decision: {_quant_decision(row[13])}")
        print(f"entry_trade_date: {row[2] or '-'}")
        print(f"entry_rule: {metadata.get('entry_rule', '-')}")
        print(f"entry_close: {_format_number(row[3])}")
        print(f"future_return_1d: {_format_number(row[4])}")
        print(f"future_return_3d: {_format_number(row[5])}")
        print(f"future_return_5d: {_format_number(row[6])}")
        print(f"future_return_10d: {_format_number(row[7])}")
        print(f"future_max_drawdown_5d: {_format_number(row[8])}")
        print(f"future_max_drawdown_10d: {_format_number(row[9])}")
        print(f"feedback_status: {row[10]}")


if __name__ == "__main__":
    main()
