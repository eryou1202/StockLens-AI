from __future__ import annotations

import csv
import re
from pathlib import Path

from src.data.symbol_mapper import SymbolMapper


SMALL_DEMO_UNIVERSE = ["300750.SZ", "000001.SZ", "600030.SH"]


def normalize_symbols(symbols: list[str]) -> list[str]:
    result: list[str] = []
    for raw in symbols:
        for token in re.split(r"[\s,;，；]+", str(raw).strip()):
            if not token:
                continue
            normalized = SymbolMapper.normalize(token)
            if normalized not in result:
                result.append(normalized)
    return result


def load_symbols_from_file(path: str) -> list[str]:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"找不到股票池文件：{source}")
    if source.suffix.lower() == ".csv":
        with source.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if rows:
            keys = list(rows[0].keys())
            preferred = next(
                (key for key in ("symbol", "stock_code", "code", "股票代码") if key in keys),
                keys[0],
            )
            return normalize_symbols([str(row.get(preferred) or "") for row in rows])
        return []
    return normalize_symbols(source.read_text(encoding="utf-8-sig").splitlines())
