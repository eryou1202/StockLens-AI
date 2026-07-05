from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from src.config.settings import load_settings


TRAINING_THRESHOLD = 30


def main() -> None:
    settings = load_settings()
    database_path = Path(settings.database_path)
    dataset_path = Path("data/ml_dataset.csv")

    signal_count = 0
    feedback_count = 0
    status_counts = {status: 0 for status in ("pending", "partial", "complete", "failed")}
    database_error: str | None = None

    if database_path.exists():
        try:
            connection = sqlite3.connect(database_path)
            tables = {
                row[0]
                for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
            }
            if "signal_snapshots" in tables:
                signal_count = int(
                    connection.execute("SELECT COUNT(*) FROM signal_snapshots").fetchone()[0]
                )
            if "signal_feedback" in tables:
                feedback_count = int(
                    connection.execute("SELECT COUNT(*) FROM signal_feedback").fetchone()[0]
                )
                for status, count in connection.execute(
                    "SELECT feedback_status, COUNT(*) FROM signal_feedback GROUP BY feedback_status"
                ):
                    if status in status_counts:
                        status_counts[status] = int(count)
        except sqlite3.Error as exc:
            database_error = str(exc)
        finally:
            if "connection" in locals():
                connection.close()

    dataset_complete_samples = 0
    dataset_error: str | None = None
    if dataset_path.exists():
        try:
            frame = pd.read_csv(dataset_path, usecols=lambda name: name == "feedback_status")
            if "feedback_status" in frame:
                status = frame["feedback_status"].astype("string").str.strip().str.lower()
                dataset_complete_samples = int((status == "complete").sum())
        except Exception as exc:
            dataset_error = f"{type(exc).__name__}: {exc}"

    print("StockLens Dataset Status\n")
    print(f"database: {database_path}")
    print(f"signal_snapshots: {signal_count}")
    print(f"signal_feedback: {feedback_count}")
    for status in ("pending", "partial", "complete", "failed"):
        print(f"{status}: {status_counts[status]}")
    print(f"ml_dataset.csv exists: {dataset_path.exists()}")
    print(f"complete samples in dataset: {dataset_complete_samples}")
    print(f"training threshold: {TRAINING_THRESHOLD}")
    print(f"training ready: {dataset_complete_samples >= TRAINING_THRESHOLD}")
    if database_error:
        print(f"database warning: {database_error}")
    if dataset_error:
        print(f"dataset warning: {dataset_error}")


if __name__ == "__main__":
    main()
