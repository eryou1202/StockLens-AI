from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from src.data.symbol_mapper import SymbolMapper
from src.data.symbol_name_resolver import SymbolNameResolver
from src.models.signal_package import StockLensSignalPackage


class CandidatePoolEditor:
    """Safely edits candidates while preserving Signal Package v1.0."""

    def __init__(self, file_path: str | Path = "data/ai_candidates.json",
                 name_resolver: SymbolNameResolver | None = None) -> None:
        self.file_path = Path(file_path)
        self.name_resolver = name_resolver or SymbolNameResolver(
            "data/signals.sqlite", str(self.file_path)
        )

    def load(self) -> StockLensSignalPackage:
        payload = json.loads(self.file_path.read_text(encoding="utf-8"))
        return StockLensSignalPackage.model_validate(payload)

    def rows(self) -> list[dict[str, Any]]:
        package = self.load()
        rows = []
        for item in package.candidates:
            metadata = item.metadata or {}
            rows.append({
                "symbol": item.stock_code,
                "stock_name": item.stock_name or "未知名称",
                "source_type": metadata.get("source_type", "ai_signal"),
                "ai_view / sentiment": item.events[0].sentiment_score if item.events else 0.0,
                "event_summary": item.events[0].summary if item.events else item.candidate_reason,
                "expected_horizon_days": item.expected_horizon_days,
                "manual_notes": metadata.get("notes", ""),
            })
        return rows

    def add_manual_watch(self, symbol: str, stock_name: str | None = None,
                         notes: str | None = None,
                         expected_horizon_days: list[int] | None = None,
                         metadata: dict[str, Any] | None = None) -> None:
        package = self.load()
        normalized = SymbolMapper.normalize(symbol)
        if any(item.stock_code == normalized for item in package.candidates):
            raise ValueError(f"{normalized} 已在候选池中。")
        now = datetime.now().astimezone()
        token = normalized.replace(".", "_")
        exchange = normalized.split(".")[-1]
        note = (notes or "").strip()
        resolved_name = (stock_name or "").strip() or self.name_resolver.resolve(normalized) or ""
        candidate = {
            "stock_code": normalized,
            "stock_name": resolved_name,
            "market": "A股",
            "exchange": exchange,
            "candidate_reason": note or "用户手动加入观察池",
            "ai_overall_score": 0.0,
            "ai_confidence": 0.2,
            "expected_horizon_days": expected_horizon_days or [3, 5],
            "events": [{
                "event_id": f"manual_event_{token}_{int(now.timestamp())}",
                "event_type": "unknown",
                "event_time": now.isoformat(),
                "publish_time": now.isoformat(),
                "title": "人工观察股",
                "summary": "用户手动加入观察池",
                "evidence_excerpt": note or None,
                "sentiment_score": 0.0,
                "event_strength": 0.0,
                "certainty": 0.0,
                "freshness_score": 0.0,
                "directness_score": 0.0,
                "source_ids": [f"manual_source_{token}"],
                "positive_points": [],
                "negative_points": ["人工观察股，不代表 AI 信息面推荐"],
            }],
            "sources": [{
                "source_id": f"manual_source_{token}",
                "source_type": "unknown",
                "source_name": "StockLens 用户",
                "title": "人工加入观察池",
                "publish_time": now.isoformat(),
                "crawl_time": now.isoformat(),
                "source_tier": 5,
                "source_confidence": 0.0,
                "is_official": False,
                "is_duplicate": False,
                "content_summary": note or "人工观察",
            }],
            "risk_flags": [{
                "risk_type": "unknown",
                "risk_level": "medium",
                "description": "人工观察股，不代表 AI 信息面推荐",
                "related_event_ids": [],
            }],
            "contradictions": [],
            "quant_focus": [{
                "focus_type": "check_trend",
                "reason": "人工观察股仅使用量化趋势与风险作辅助跟踪。",
            }],
            "metadata": {
                **(metadata or {}),
                "source_type": "manual_watch",
                "event_type": "manual_watch",
                "manual_review_required": True,
                "notes": note,
            },
        }
        payload = package.model_dump(mode="json")
        payload["candidates"].append(candidate)
        self._write_validated(payload)

    def remove(self, symbol: str) -> int:
        package = self.load()
        normalized = SymbolMapper.normalize(symbol)
        payload = package.model_dump(mode="json")
        before = len(payload["candidates"])
        payload["candidates"] = [
            item for item in payload["candidates"] if item["stock_code"] != normalized
        ]
        removed = before - len(payload["candidates"])
        if not removed:
            return 0
        if not payload["candidates"]:
            raise ValueError("Signal Package v1.0 至少需要一个候选股，不能删除最后一只。")
        self._write_validated(payload)
        return removed

    def save_json(self, raw: str | bytes | dict[str, Any]) -> None:
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8-sig")
        payload = json.loads(raw) if isinstance(raw, str) else raw
        self._write_validated(payload)

    def _write_validated(self, payload: dict[str, Any]) -> None:
        validated = StockLensSignalPackage.model_validate(payload)
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self.file_path.write_text(
            json.dumps(validated.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
