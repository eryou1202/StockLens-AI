from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.audit.audit_metrics import AuditMetricsBuilder
from src.audit.audit_schema import AuditRequest, AuditSample, AuditSummary


class AuditStore:
    def __init__(self, database_path: str = "data/audit/algorithm_audit.sqlite") -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _init_db(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS audit_runs (
                    audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    audit_name TEXT,
                    mode TEXT NOT NULL,
                    start_date TEXT NOT NULL,
                    end_date TEXT NOT NULL,
                    step_days INTEGER NOT NULL,
                    lookback_days INTEGER NOT NULL,
                    symbols_count INTEGER NOT NULL,
                    samples_count INTEGER NOT NULL DEFAULT 0,
                    complete_samples INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    metadata_json TEXT
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS audit_samples (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    audit_id INTEGER NOT NULL,
                    symbol TEXT NOT NULL,
                    stock_name TEXT,
                    as_of_time TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    action TEXT,
                    action_level TEXT,
                    final_score REAL,
                    quant_score REAL,
                    quant_decision TEXT,
                    ai_view TEXT,
                    current_price REAL,
                    future_return_1d REAL,
                    future_return_3d REAL,
                    future_return_5d REAL,
                    future_return_10d REAL,
                    future_max_drawdown_5d REAL,
                    future_max_drawdown_10d REAL,
                    trend_score REAL,
                    momentum_score REAL,
                    volume_score REAL,
                    risk_score REAL,
                    overheat_score REAL,
                    macd_score REAL,
                    return_5d REAL,
                    return_20d REAL,
                    close_ma20_gap REAL,
                    ma5_ma20_gap REAL,
                    ma20_ma60_gap REAL,
                    rsi_14 REAL,
                    macd_hist REAL,
                    volume_ratio_5d REAL,
                    max_drawdown_20d REAL,
                    volatility_20d REAL,
                    sample_note TEXT,
                    error_message TEXT,
                    FOREIGN KEY(audit_id) REFERENCES audit_runs(audit_id)
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS ix_audit_samples_audit_symbol_time "
                "ON audit_samples(audit_id, symbol, as_of_time)"
            )

    def create_run(self, request: AuditRequest) -> int:
        now = datetime.now().isoformat()
        symbols = request.symbols[: request.max_symbols] if request.max_symbols else request.symbols
        metadata = {
            "request": request.model_dump(mode="json"),
            "isolation": {
                "production_database_used": False,
                "recommendation_tracking_written": False,
                "signal_feedback_written": False,
                "signal_snapshots_written": False,
                "positions_written": False,
                "candidate_file_written": False,
            },
        }
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO audit_runs (
                    audit_name, mode, start_date, end_date, step_days, lookback_days,
                    symbols_count, samples_count, complete_samples, created_at, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 0, 0, ?, ?)
                """,
                (
                    request.audit_name,
                    request.mode.value,
                    request.start_date.isoformat(),
                    request.end_date.isoformat(),
                    request.step_days,
                    request.lookback_days,
                    len(symbols),
                    now,
                    json.dumps(metadata, ensure_ascii=False, default=str),
                ),
            )
            return int(cursor.lastrowid)

    def save_samples(self, samples: list[AuditSample]) -> int:
        if not samples:
            return 0
        fields = [
            "audit_id", "symbol", "stock_name", "as_of_time", "mode", "action",
            "action_level", "final_score", "quant_score", "quant_decision", "ai_view",
            "current_price", "future_return_1d", "future_return_3d", "future_return_5d",
            "future_return_10d", "future_max_drawdown_5d", "future_max_drawdown_10d",
            "trend_score", "momentum_score", "volume_score", "risk_score", "overheat_score",
            "macd_score", "return_5d", "return_20d", "close_ma20_gap", "ma5_ma20_gap",
            "ma20_ma60_gap", "rsi_14", "macd_hist", "volume_ratio_5d",
            "max_drawdown_20d", "volatility_20d", "sample_note", "error_message",
        ]
        rows: list[tuple[Any, ...]] = []
        for sample in samples:
            payload = sample.model_dump(mode="json")
            rows.append(tuple(payload.get(field) for field in fields))
        placeholders = ", ".join("?" for _ in fields)
        with self._connect() as connection:
            connection.executemany(
                f"INSERT INTO audit_samples ({', '.join(fields)}) VALUES ({placeholders})",
                rows,
            )
        return len(rows)

    def finalize_run(self, audit_id: int, summary: dict[str, Any]) -> None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT metadata_json FROM audit_runs WHERE audit_id=?", (audit_id,)
            ).fetchone()
            if row is None:
                raise ValueError(f"找不到 audit_id={audit_id}")
            try:
                metadata = json.loads(row["metadata_json"] or "{}")
            except json.JSONDecodeError:
                metadata = {}
            metadata["summary"] = summary
            connection.execute(
                """
                UPDATE audit_runs
                SET samples_count=?, complete_samples=?, metadata_json=?
                WHERE audit_id=?
                """,
                (
                    int(summary.get("samples_count", 0)),
                    int(summary.get("complete_samples", 0)),
                    json.dumps(metadata, ensure_ascii=False, default=str),
                    audit_id,
                ),
            )

    def resolve_audit_id(self, audit_id: str | int = "latest") -> int:
        with self._connect() as connection:
            if str(audit_id).lower() == "latest":
                row = connection.execute(
                    "SELECT audit_id FROM audit_runs ORDER BY audit_id DESC LIMIT 1"
                ).fetchone()
            else:
                row = connection.execute(
                    "SELECT audit_id FROM audit_runs WHERE audit_id=?", (int(audit_id),)
                ).fetchone()
        if row is None:
            raise ValueError(f"找不到 audit_id={audit_id}")
        return int(row["audit_id"])

    def load_samples(self, audit_id: str | int = "latest") -> list[AuditSample]:
        resolved = self.resolve_audit_id(audit_id)
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM audit_samples WHERE audit_id=? ORDER BY symbol, as_of_time",
                (resolved,),
            ).fetchall()
        fields = set(AuditSample.model_fields)
        return [
            AuditSample.model_validate({key: row[key] for key in fields}) for row in rows
        ]

    def load_summary(self, audit_id: str | int = "latest") -> AuditSummary:
        resolved = self.resolve_audit_id(audit_id)
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM audit_runs WHERE audit_id=?", (resolved,)
            ).fetchone()
        if row is None:
            raise ValueError(f"找不到 audit_id={resolved}")
        try:
            metadata = json.loads(row["metadata_json"] or "{}")
        except json.JSONDecodeError:
            metadata = {}
        metrics = metadata.get("summary") or AuditMetricsBuilder.build_summary(
            self.load_samples(resolved)
        )
        return AuditSummary(
            audit_id=resolved,
            audit_name=row["audit_name"],
            start_date=datetime.fromisoformat(row["start_date"]),
            end_date=datetime.fromisoformat(row["end_date"]),
            symbols_count=row["symbols_count"],
            samples_count=metrics.get("samples_count", row["samples_count"]),
            complete_samples=metrics.get("complete_samples", row["complete_samples"]),
            action_distribution=metrics.get("action_distribution", {}),
            action_metrics=metrics.get("action_metrics", {}),
            quant_decision_metrics=metrics.get("quant_decision_metrics", {}),
            score_future_return_corr_5d=metrics.get("score_future_return_corr_5d"),
            score_future_return_corr_10d=metrics.get("score_future_return_corr_10d"),
            final_score_future_return_corr_5d=metrics.get("final_score_future_return_corr_5d"),
            final_score_future_return_corr_10d=metrics.get("final_score_future_return_corr_10d"),
            ranking_warning=bool(metrics.get("ranking_warning", False)),
            metrics=metrics,
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    def export_csv(
        self,
        audit_id: str | int = "latest",
        output_dir: str = "data/audit",
    ) -> dict[str, str]:
        resolved = self.resolve_audit_id(audit_id)
        samples = self.load_samples(resolved)
        summary = self.load_summary(resolved)
        metrics = summary.metrics
        target = Path(output_dir)
        target.mkdir(parents=True, exist_ok=True)
        paths = {
            "summary": target / "audit_summary.csv",
            "action_metrics": target / "audit_action_metrics.csv",
            "quant_decision_metrics": target / "audit_quant_decision_metrics.csv",
            "cases": target / "audit_cases.csv",
            "samples": target / "audit_samples.csv",
        }

        summary_row = {
            "audit_id": resolved,
            "audit_name": summary.audit_name,
            "start_date": summary.start_date.isoformat(),
            "end_date": summary.end_date.isoformat(),
            "symbols_count": summary.symbols_count,
            **{
                key: value for key, value in metrics.items()
                if key not in {"action_metrics", "quant_decision_metrics", "action_distribution"}
                and not isinstance(value, (dict, list))
            },
            "action_distribution_json": json.dumps(summary.action_distribution, ensure_ascii=False),
        }
        pd.DataFrame([summary_row]).to_csv(paths["summary"], index=False, encoding="utf-8-sig")
        pd.DataFrame(cls_group_rows(resolved, "action", summary.action_metrics)).to_csv(
            paths["action_metrics"], index=False, encoding="utf-8-sig"
        )
        pd.DataFrame(cls_group_rows(resolved, "quant_decision", summary.quant_decision_metrics)).to_csv(
            paths["quant_decision_metrics"], index=False, encoding="utf-8-sig"
        )
        cases = AuditMetricsBuilder.build_cases(samples)
        case_columns = ["case_type", *AuditSample.model_fields.keys()]
        pd.DataFrame(cases, columns=case_columns).to_csv(
            paths["cases"], index=False, encoding="utf-8-sig"
        )
        pd.DataFrame([item.model_dump(mode="json") for item in samples]).to_csv(
            paths["samples"], index=False, encoding="utf-8-sig"
        )
        return {key: str(path) for key, path in paths.items()}


def cls_group_rows(audit_id: int, group_name: str, groups: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"audit_id": audit_id, group_name: name, **values}
        for name, values in groups.items()
    ]
