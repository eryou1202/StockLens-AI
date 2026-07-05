from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel


class AppSettings(BaseModel):
    database_path: str = "data/signals.sqlite"
    cache_dir: str = "data/cache"
    default_lookback_days: int = 120
    feedback_horizons: list[int] = [1, 3, 5, 10]

    market_provider: str = "mock"
    market_frequency: str = "1d"
    market_adjust_type: str = "qfq"


def load_settings(config_path: str | Path = "config/config.yaml") -> AppSettings:
    path = Path(config_path)
    if not path.exists():
        return AppSettings()

    raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    runtime = raw.get("runtime", {})
    market_data = raw.get("market_data", {})

    return AppSettings(
        database_path=runtime.get("database_path", "data/signals.sqlite"),
        cache_dir=runtime.get("cache_dir", "data/cache"),
        default_lookback_days=runtime.get("default_lookback_days", 120),
        feedback_horizons=runtime.get("feedback_horizons", [1, 3, 5, 10]),
        market_provider=market_data.get("provider", "mock"),
        market_frequency=market_data.get("frequency", "1d"),
        market_adjust_type=market_data.get("adjust_type", "qfq"),
    )
