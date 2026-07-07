from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from src.data.symbol_mapper import SymbolMapper


REFERENCE_COLUMNS = [
    "symbol", "code", "exchange", "stock_name", "asset_type", "list_status", "updated_at", "source"
]

ASSET_TYPES = {"stock", "index", "fund", "unknown"}


class AShareSymbolReference:
    """Build and search a local A-share code/name reference without serving production flow."""

    def __init__(
        self,
        reference_path: str = "data/reference/a_share_symbols.csv",
        database_path: str = "data/signals.sqlite",
        candidate_file: str = "data/ai_candidates.json",
        name_cache_file: str = "data/symbol_name_cache.json",
    ) -> None:
        self.reference_path = Path(reference_path)
        self.database_path = Path(database_path)
        self.candidate_file = Path(candidate_file)
        self.name_cache_file = Path(name_cache_file)

    def build(
        self,
        symbols: Iterable[str] | None = None,
        include_provider: bool = True,
        stock_only: bool = False,
        include_index: bool = False,
    ) -> pd.DataFrame:
        records: dict[str, dict[str, Any]] = {}
        self._merge_existing(records)
        if include_provider:
            self._merge_baostock(records)
        self._merge_candidates(records)
        self._merge_database(records)
        self._merge_name_cache(records)
        for raw in symbols or []:
            self._upsert(records, raw, None, "user_input")

        frame = pd.DataFrame(records.values(), columns=REFERENCE_COLUMNS)
        if not frame.empty:
            if stock_only:
                frame = frame.loc[frame["asset_type"] == "stock"]
            elif include_index:
                frame = frame.loc[frame["asset_type"].isin(["stock", "index"])]
            frame = frame.sort_values(["exchange", "code"], kind="stable").reset_index(drop=True)
        self.reference_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(self.reference_path, index=False, encoding="utf-8-sig")
        return frame

    def load(self) -> pd.DataFrame:
        if not self.reference_path.exists():
            return pd.DataFrame(columns=REFERENCE_COLUMNS)
        try:
            frame = pd.read_csv(self.reference_path, dtype=str, encoding="utf-8-sig")
        except (OSError, pd.errors.ParserError, UnicodeDecodeError):
            return pd.DataFrame(columns=REFERENCE_COLUMNS)
        for column in REFERENCE_COLUMNS:
            if column not in frame.columns:
                frame[column] = None
        return frame[REFERENCE_COLUMNS]

    def search(self, query: str, limit: int = 30) -> pd.DataFrame:
        frame = self.load()
        text = str(query or "").strip()
        if frame.empty or not text:
            return frame.head(0)
        folded = text.casefold()
        masks = []
        for column in ("symbol", "code", "stock_name"):
            masks.append(frame[column].fillna("").astype(str).str.casefold().str.contains(folded, regex=False))
        mask = masks[0] | masks[1] | masks[2]
        result = frame.loc[mask].copy()
        exact = (
            result["symbol"].fillna("").str.casefold().eq(folded)
            | result["code"].fillna("").str.casefold().eq(folded)
            | result["stock_name"].fillna("").str.casefold().eq(folded)
        )
        result["_exact"] = exact.astype(int)
        return result.sort_values(["_exact", "symbol"], ascending=[False, True]).drop(
            columns="_exact"
        ).head(max(1, int(limit)))

    def name_map(self, asset_type: str | None = None) -> dict[str, str]:
        frame = self.load()
        if asset_type is not None:
            frame = frame.loc[frame["asset_type"] == asset_type]
        result: dict[str, str] = {}
        for row in frame.to_dict("records"):
            name = self._clean_name(row.get("stock_name"))
            if name:
                result[str(row["symbol"])] = name
        return result

    def symbols(self, asset_type: str = "stock") -> list[str]:
        """Return a universe that is stock-only by default."""
        if asset_type not in ASSET_TYPES:
            raise ValueError(f"unsupported asset_type: {asset_type}")
        frame = self.load()
        return frame.loc[frame["asset_type"] == asset_type, "symbol"].dropna().astype(str).tolist()

    def asset_type_map(self) -> dict[str, str]:
        frame = self.load()
        result: dict[str, str] = {}
        for row in frame.to_dict("records"):
            if not row.get("symbol"):
                continue
            asset_type = str(row.get("asset_type") or "unknown").strip().lower()
            result[str(row["symbol"])] = asset_type if asset_type in ASSET_TYPES else "unknown"
        return result

    def _merge_existing(self, records: dict[str, dict[str, Any]]) -> None:
        for row in self.load().to_dict("records"):
            self._upsert(
                records,
                row.get("symbol"),
                row.get("stock_name"),
                row.get("source") or "existing_reference",
                row.get("asset_type"),
                row.get("list_status") or "unknown",
                row.get("updated_at"),
            )

    def _merge_candidates(self, records: dict[str, dict[str, Any]]) -> None:
        if not self.candidate_file.exists():
            return
        try:
            payload = json.loads(self.candidate_file.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            return
        candidates = payload.get("candidates", []) if isinstance(payload, dict) else []
        for candidate in candidates:
            if isinstance(candidate, dict):
                self._upsert(
                    records,
                    candidate.get("stock_code") or candidate.get("symbol"),
                    candidate.get("stock_name"),
                    "ai_candidates",
                )

    def _merge_database(self, records: dict[str, dict[str, Any]]) -> None:
        if not self.database_path.exists():
            return
        try:
            with sqlite3.connect(self.database_path) as connection:
                for table in ("positions", "recommendation_tracking"):
                    exists = connection.execute(
                        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
                    ).fetchone()
                    if not exists:
                        continue
                    rows = connection.execute(
                        f"SELECT symbol, stock_name FROM {table} "
                        "WHERE symbol IS NOT NULL ORDER BY rowid DESC"
                    ).fetchall()
                    for symbol, name in rows:
                        self._upsert(records, symbol, name, table)
        except sqlite3.Error:
            return

    def _merge_name_cache(self, records: dict[str, dict[str, Any]]) -> None:
        if not self.name_cache_file.exists():
            return
        try:
            payload = json.loads(self.name_cache_file.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            return
        if not isinstance(payload, dict):
            return
        for symbol, value in payload.items():
            name = value.get("name") if isinstance(value, dict) else value
            self._upsert(records, symbol, name, "symbol_name_cache")

    def _merge_baostock(self, records: dict[str, dict[str, Any]]) -> None:
        try:
            import baostock as bs
        except ImportError:
            return
        login = None
        try:
            login = bs.login()
            if login.error_code != "0":
                return
            day = datetime.now()
            result = None
            for _ in range(15):
                if day.weekday() < 5:
                    candidate = bs.query_all_stock(day.strftime("%Y-%m-%d"))
                    if candidate.error_code == "0":
                        result = candidate
                        break
                day -= timedelta(days=1)
            if result is None:
                return
            while result.next():
                row = dict(zip(result.fields, result.get_row_data()))
                code = row.get("code")
                if not code or not str(code).lower().startswith(("sh.", "sz.", "bj.")):
                    continue
                self._upsert(
                    records,
                    code,
                    row.get("code_name") or row.get("codeName"),
                    "baostock",
                    None,
                    "listed" if str(row.get("tradeStatus", "1")) == "1" else "suspended",
                )
        except Exception:
            return
        finally:
            if login is not None:
                try:
                    bs.logout()
                except Exception:
                    pass

    @classmethod
    def _upsert(
        cls,
        records: dict[str, dict[str, Any]],
        raw_symbol: Any,
        raw_name: Any,
        source: Any,
        asset_type: Any = None,
        list_status: Any = "unknown",
        updated_at: Any = None,
    ) -> None:
        if raw_symbol is None:
            return
        try:
            symbol = SymbolMapper.normalize(str(raw_symbol))
        except (TypeError, ValueError):
            return
        code, exchange = symbol.split(".")
        name = cls._clean_name(raw_name)
        normalized_asset_type = cls._normalize_asset_type(asset_type, code, exchange)
        existing = records.get(symbol)
        now = str(updated_at or datetime.now().isoformat(timespec="seconds"))
        if existing is None:
            records[symbol] = {
                "symbol": symbol,
                "code": code,
                "exchange": exchange,
                "stock_name": name,
                "asset_type": normalized_asset_type,
                "list_status": str(list_status or "unknown"),
                "updated_at": now,
                "source": str(source or "unknown"),
            }
            return
        sources = [item for item in str(existing.get("source") or "").split("+") if item]
        source_text = str(source or "unknown")
        if source_text not in sources:
            sources.append(source_text)
        existing["source"] = "+".join(sources)
        if name and not cls._clean_name(existing.get("stock_name")):
            existing["stock_name"] = name
        if existing.get("asset_type") not in ASSET_TYPES or existing.get("asset_type") == "unknown":
            existing["asset_type"] = normalized_asset_type
        if str(list_status) not in {"", "unknown", "None", "nan"}:
            existing["list_status"] = str(list_status)
        existing["updated_at"] = now

    @staticmethod
    def _clean_name(value: Any) -> str | None:
        if value is None or pd.isna(value):
            return None
        text = str(value).strip()
        return None if text in {"", "-", "未知名称", "None", "nan"} else text

    @staticmethod
    def _normalize_asset_type(value: Any, code: str, exchange: str) -> str:
        normalized = str(value or "").strip().lower()
        if normalized in ASSET_TYPES:
            return normalized
        if exchange == "SH":
            if code.startswith(("600", "601", "603", "605", "688")):
                return "stock"
            if code.startswith("000"):
                return "index"
            if code.startswith("5"):
                return "fund"
        elif exchange == "SZ":
            if code.startswith(("000", "001", "002", "003", "300", "301")):
                return "stock"
            if code.startswith("399"):
                return "index"
            if code.startswith(("15", "16", "18")):
                return "fund"
        elif exchange == "BJ" and code.startswith(("4", "8", "9")):
            return "stock"
        return "unknown"
