from __future__ import annotations

import ast
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.models.schemas import MarketBar, MarketDataBundle


class MarketDataCache:
    """
    简单 CSV 缓存。

    目的：
    - 避免每次重复请求 Baostock / AKShare
    - 保留原始 provider 字段转换后的统一结果

    TODO:
    - 改成 SQLite / DuckDB / Parquet
    - 增量更新
    - 校验缓存是否覆盖请求区间
    - 加入数据版本
    """

    def __init__(self, cache_dir: str = "data/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, provider: str, symbol: str, frequency: str, adjust_type: str) -> Path:
        safe_symbol = symbol.replace(".", "_")
        return self.cache_dir / provider / frequency / adjust_type / f"{safe_symbol}.csv"

    def save_bundle(self, bundle: MarketDataBundle) -> None:
        path = self._cache_path(bundle.provider, bundle.symbol, bundle.frequency, bundle.adjust_type)
        path.parent.mkdir(parents=True, exist_ok=True)

        rows = [bar.model_dump(mode="json") for bar in bundle.bars]
        df = pd.DataFrame(rows)
        if df.empty:
            return

        df.to_csv(path, index=False, encoding="utf-8-sig")

    def load_bundle(
        self,
        provider: str,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
        frequency: str,
        adjust_type: str,
    ) -> MarketDataBundle | None:
        path = self._cache_path(provider, symbol, frequency, adjust_type)
        if not path.exists():
            return None

        df = pd.read_csv(path)
        if df.empty:
            return None

        if "trade_time" not in df.columns:
            return None

        df["trade_time"] = pd.to_datetime(df["trade_time"])

        mask = (df["trade_time"] >= pd.Timestamp(start_time)) & (
            df["trade_time"] <= pd.Timestamp(end_time)
        )
        sliced = df.loc[mask].copy()

        if sliced.empty:
            return None

        bars: list[MarketBar] = []
        for _, row in sliced.iterrows():
            raw_row = row.to_dict()
            cleaned = self._clean_market_bar_row(raw_row)
            bars.append(MarketBar(**cleaned))

        return MarketDataBundle(
            symbol=symbol,
            start_time=start_time,
            end_time=end_time,
            frequency=frequency,
            adjust_type=adjust_type,
            provider=provider,
            bars=bars,
            data_quality={
                "from_cache": True,
                "rows": len(bars),
                "cache_path": str(path),
            },
        )

    def _clean_market_bar_row(self, row: dict[str, Any]) -> dict[str, Any]:
        """
        修复 CSV 读取后产生的类型问题。

        典型问题：
        - trade_status: "1" 保存后被读成 int 1
        - raw: dict 保存后被读成字符串 "{...}"
        - 空值被读成 NaN
        """
        cleaned: dict[str, Any] = {}

        for key, value in row.items():
            cleaned[key] = self._nan_to_none(value)

        # trade_time / fetched_at 转 datetime
        if cleaned.get("trade_time") is not None:
            cleaned["trade_time"] = pd.to_datetime(cleaned["trade_time"]).to_pydatetime()

        if cleaned.get("fetched_at") is not None:
            cleaned["fetched_at"] = pd.to_datetime(cleaned["fetched_at"]).to_pydatetime()

        # trade_status 必须是 str 或 None
        if cleaned.get("trade_status") is not None:
            cleaned["trade_status"] = str(cleaned["trade_status"])

        # raw 必须是 dict
        cleaned["raw"] = self._parse_raw_dict(cleaned.get("raw"))

        # is_st 可能被读成 True/False、0/1、"0"/"1"
        cleaned["is_st"] = self._parse_bool(cleaned.get("is_st"))

        return cleaned

    @staticmethod
    def _nan_to_none(value: Any) -> Any:
        try:
            if pd.isna(value):
                return None
        except TypeError:
            pass
        return value

    @staticmethod
    def _parse_raw_dict(value: Any) -> dict[str, Any]:
        if value is None:
            return {}

        if isinstance(value, dict):
            return value

        if isinstance(value, str):
            text = value.strip()
            if not text:
                return {}

            try:
                parsed = ast.literal_eval(text)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                return {"raw_text": text}

        return {}

    @staticmethod
    def _parse_bool(value: Any) -> bool | None:
        if value is None:
            return None

        if isinstance(value, bool):
            return value

        text = str(value).strip().lower()

        if text in {"1", "true", "yes"}:
            return True

        if text in {"0", "false", "no"}:
            return False

        return None