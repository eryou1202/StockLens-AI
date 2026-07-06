from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from src.data.symbol_mapper import SymbolMapper
from src.models.signal_package import StockLensSignalPackage


class SymbolNameResolver:
    """Resolve A-share names from local sources first, then Baostock as a safe fallback."""

    INVALID_NAMES = {"", "-", "未知名称", "None", "null"}

    def __init__(self, database_path: str, candidate_file: str = "data/ai_candidates.json"):
        self.database_path = Path(database_path)
        self.candidate_file = Path(candidate_file)
        self.cache_file = self.candidate_file.parent / "symbol_name_cache.json"

    def resolve(self, symbol: str) -> str | None:
        normalized = SymbolMapper.normalize(symbol)
        name = (
            self._from_candidates(normalized)
            or self._from_positions(normalized)
            or self._from_cache(normalized)
            or self._from_baostock(normalized)
        )
        name = self._clean_name(name)
        if name:
            self._save_cache_name(normalized, name)
        return name

    def resolve_many(self, symbols: list[str]) -> dict[str, str]:
        result: dict[str, str] = {}
        for symbol in dict.fromkeys(symbols):
            try:
                normalized = SymbolMapper.normalize(symbol)
                name = self.resolve(normalized)
            except Exception:
                continue
            if name:
                result[normalized] = name
        return result

    def update_position_name_if_missing(self, symbol: str) -> str | None:
        normalized = SymbolMapper.normalize(symbol)
        name = self.resolve(normalized)
        if not name or not self.database_path.exists():
            return name
        now = datetime.now().isoformat()
        with sqlite3.connect(self.database_path) as connection:
            connection.execute(
                """
                UPDATE positions SET stock_name=?, updated_at=?
                WHERE symbol=? AND (stock_name IS NULL OR TRIM(stock_name)='' OR stock_name='-')
                """,
                (name, now, normalized),
            )
        return name

    def update_tracking_name_if_missing(self, symbol: str) -> str | None:
        normalized = SymbolMapper.normalize(symbol)
        name = self.resolve(normalized)
        if not name or not self.database_path.exists():
            return name
        now = datetime.now().isoformat()
        with sqlite3.connect(self.database_path) as connection:
            if not self._table_exists(connection, "recommendation_tracking"):
                return name
            connection.execute(
                """
                UPDATE recommendation_tracking SET stock_name=?, updated_at=?
                WHERE symbol=? AND (stock_name IS NULL OR TRIM(stock_name)='' OR stock_name='-')
                """,
                (name, now, normalized),
            )
        return name

    def update_candidate_name_if_missing(self, symbol: str) -> str | None:
        normalized = SymbolMapper.normalize(symbol)
        name = self.resolve(normalized)
        if not name or not self.candidate_file.exists():
            return name
        payload = json.loads(self.candidate_file.read_text(encoding="utf-8"))
        changed = False
        for candidate in payload.get("candidates", []):
            if candidate.get("stock_code") == normalized and not self._clean_name(candidate.get("stock_name")):
                candidate["stock_name"] = name
                changed = True
        if changed:
            validated = StockLensSignalPackage.model_validate(payload)
            self._atomic_write(self.candidate_file, validated.model_dump(mode="json"))
        return name

    def scan_symbols(self) -> list[str]:
        symbols: set[str] = set()
        if self.candidate_file.exists():
            try:
                package = StockLensSignalPackage.model_validate_json(
                    self.candidate_file.read_text(encoding="utf-8")
                )
                symbols.update(item.stock_code for item in package.candidates)
            except Exception:
                pass
        if self.database_path.exists():
            with sqlite3.connect(self.database_path) as connection:
                for table in ("positions", "recommendation_tracking"):
                    if self._table_exists(connection, table):
                        symbols.update(
                            str(row[0]) for row in connection.execute(
                                f"SELECT DISTINCT symbol FROM {table} WHERE symbol IS NOT NULL"
                            )
                        )
        normalized: list[str] = []
        for symbol in sorted(symbols):
            try:
                normalized.append(SymbolMapper.normalize(symbol))
            except Exception:
                continue
        return normalized

    def backfill_all(self) -> dict[str, Any]:
        symbols = self.scan_symbols()
        success: list[str] = []
        failed: list[str] = []
        for symbol in symbols:
            try:
                name = self.resolve(symbol)
                if not name:
                    failed.append(symbol)
                    continue
                self.update_position_name_if_missing(symbol)
                self.update_tracking_name_if_missing(symbol)
                self.update_candidate_name_if_missing(symbol)
                success.append(symbol)
            except Exception:
                failed.append(symbol)
        return {
            "total": len(symbols),
            "success": len(success),
            "failed": len(failed),
            "success_symbols": success,
            "failed_symbols": failed,
        }

    def _from_candidates(self, symbol: str) -> str | None:
        if not self.candidate_file.exists():
            return None
        try:
            package = StockLensSignalPackage.model_validate_json(
                self.candidate_file.read_text(encoding="utf-8")
            )
            for candidate in package.candidates:
                if candidate.stock_code == symbol:
                    return self._clean_name(candidate.stock_name)
        except Exception:
            return None
        return None

    def _from_positions(self, symbol: str) -> str | None:
        if not self.database_path.exists():
            return None
        try:
            with sqlite3.connect(self.database_path) as connection:
                if not self._table_exists(connection, "positions"):
                    return None
                rows = connection.execute(
                    """
                    SELECT stock_name FROM positions
                    WHERE symbol=? AND stock_name IS NOT NULL AND TRIM(stock_name) NOT IN ('', '-')
                    ORDER BY id DESC
                    """,
                    (symbol,),
                ).fetchall()
            for row in rows:
                name = self._clean_name(row[0])
                if name:
                    return name
        except sqlite3.Error:
            return None
        return None

    def _from_cache(self, symbol: str) -> str | None:
        cache = self._load_cache()
        value = cache.get(symbol)
        if isinstance(value, dict):
            value = value.get("name")
        return self._clean_name(value)

    @staticmethod
    def _from_baostock(symbol: str) -> str | None:
        try:
            import baostock as bs
            login = bs.login()
            if login.error_code != "0":
                return None
            try:
                result = bs.query_stock_basic(code=SymbolMapper.to_baostock(symbol))
                if result.error_code != "0" or not result.next():
                    return None
                row = dict(zip(result.fields, result.get_row_data()))
                return SymbolNameResolver._clean_name(
                    row.get("code_name") or row.get("codeName") or row.get("name")
                )
            finally:
                bs.logout()
        except Exception:
            return None

    def _load_cache(self) -> dict[str, Any]:
        if not self.cache_file.exists():
            return {}
        try:
            data = json.loads(self.cache_file.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    def _save_cache_name(self, symbol: str, name: str) -> None:
        cache = self._load_cache()
        existing = cache.get(symbol)
        if isinstance(existing, dict) and existing.get("name") == name:
            return
        cache[symbol] = {"name": name, "updated_at": datetime.now().isoformat()}
        self._atomic_write(self.cache_file, cache)

    @staticmethod
    def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)

    @classmethod
    def _clean_name(cls, value: Any) -> str | None:
        if value is None:
            return None
        name = str(value).strip()
        return None if name in cls.INVALID_NAMES else name

    @staticmethod
    def _table_exists(connection: sqlite3.Connection, table: str) -> bool:
        row = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
        ).fetchone()
        return row is not None
