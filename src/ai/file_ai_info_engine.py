from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from src.ai.ai_info_engine import AIInfoEngine
from src.models.schemas import AICandidate, FeedbackSummary
from src.models.signal_package import SignalCandidate, StockLensSignalPackage


class FileAIInfoEngine(AIInfoEngine):
    """
    从 JSON 文件读取 StockLens Signal Package v1.0。

    输入文件：
        data/ai_candidates.json

    作用：
        StockLens Signal Package v1.0
        ↓
        校验格式
        ↓
        转换成项目内部 AICandidate
        ↓
        交给 Baostock + QuantEngine
    """

    def __init__(self, file_path: str = "data/ai_candidates.json"):
        self.file_path = Path(file_path)

    def generate_candidates(self, as_of_time: datetime | None = None) -> list[AICandidate]:
        package = self._load_and_validate_package()

        final_as_of_time = as_of_time or package.run_context.as_of_time

        return [
            self._convert_candidate(candidate, final_as_of_time, package)
            for candidate in package.candidates
        ]

    def receive_feedback(self, summary: FeedbackSummary) -> None:
        """
        TODO:
        - 未来可以把反馈摘要写入 data/feedback_summary.json
        - 再由另一个 ChatGPT 窗口读取反馈摘要，改进候选池搜索
        """
        return None

    def _load_and_validate_package(self) -> StockLensSignalPackage:
        if not self.file_path.exists():
            raise FileNotFoundError(f"找不到 AI 候选池文件: {self.file_path}")

        text = self.file_path.read_text(encoding="utf-8")

        try:
            raw = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"AI 候选池 JSON 格式错误: {exc}") from exc

        try:
            return StockLensSignalPackage.model_validate(raw)
        except ValidationError as exc:
            raise ValueError(
                "AI 候选池不符合 StockLens Signal Package v1.0：\n"
                + str(exc)
            ) from exc

    def _convert_candidate(
        self,
        candidate: SignalCandidate,
        as_of_time: datetime,
        package: StockLensSignalPackage,
    ) -> AICandidate:
        main_event = candidate.events[0]

        sentiment_score = self._average([event.sentiment_score for event in candidate.events])
        event_strength = self._average([event.event_strength for event in candidate.events])
        source_confidence = self._average(
            [source.source_confidence for source in candidate.sources if not source.is_duplicate]
        )

        ai_confidence = candidate.ai_confidence

        # 信息冲突会降低 AI 置信度
        for contradiction in candidate.contradictions:
            if contradiction.impact_on_confidence < 0:
                ai_confidence += contradiction.impact_on_confidence

        ai_confidence = max(0.0, min(1.0, ai_confidence))

        risk_texts = [
            f"[{risk.risk_level}] {risk.risk_type}: {risk.description}"
            for risk in candidate.risk_flags
        ]

        source_urls = [
            source.url for source in candidate.sources if source.url
        ]

        raw_evidence = self._collect_evidence(candidate)

        metadata: dict[str, Any] = {
            "schema_version": package.schema_version,
            "run_context": package.run_context.model_dump(mode="json"),
            "signal_candidate": candidate.model_dump(mode="json"),
            "candidate_reason": candidate.candidate_reason,
            "ai_overall_score": candidate.ai_overall_score,
            "market": candidate.market,
            "exchange": candidate.exchange,
            "industry": candidate.industry,
            "sector": candidate.sector,
            "concept_tags": candidate.concept_tags,
            "expected_horizon_days": candidate.expected_horizon_days,
            "events": [event.model_dump(mode="json") for event in candidate.events],
            "sources": [source.model_dump(mode="json") for source in candidate.sources],
            "risk_flags": [risk.model_dump(mode="json") for risk in candidate.risk_flags],
            "contradictions": [
                item.model_dump(mode="json") for item in candidate.contradictions
            ],
            "quant_focus": [
                item.model_dump(mode="json") for item in candidate.quant_focus
            ],
            "candidate_metadata": candidate.metadata,
            "source_type": candidate.metadata.get("source_type", "ai_signal"),
            "manual_review_required": candidate.metadata.get("manual_review_required", False),
            "manual_notes": candidate.metadata.get("notes"),
        }

        return AICandidate(
            stock_code=candidate.stock_code,
            stock_name=candidate.stock_name,
            as_of_time=as_of_time,
            event_time=main_event.event_time or main_event.publish_time or as_of_time,
            event_type=main_event.event_type,
            event_summary=main_event.summary,
            sentiment_score=sentiment_score,
            event_strength=event_strength,
            source_confidence=source_confidence,
            ai_confidence=ai_confidence,
            risk_flags=risk_texts,
            source_urls=source_urls,
            raw_evidence=raw_evidence,
            metadata=metadata,
        )

    @staticmethod
    def _average(values: list[float], default: float = 0.0) -> float:
        if not values:
            return default
        return sum(values) / len(values)

    @staticmethod
    def _collect_evidence(candidate: SignalCandidate) -> str:
        parts: list[str] = []

        for event in candidate.events:
            parts.append(f"事件：{event.title}")
            parts.append(f"摘要：{event.summary}")
            if event.evidence_excerpt:
                parts.append(f"证据：{event.evidence_excerpt}")

        for source in candidate.sources:
            parts.append(f"来源：{source.title}")
            if source.key_quote:
                parts.append(f"引用：{source.key_quote}")

        return "\n".join(parts)
