from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


EventType = Literal[
    "earnings_forecast",
    "earnings_report",
    "policy_benefit",
    "policy_risk",
    "industry_hotspot",
    "concept_hotspot",
    "major_contract",
    "product_release",
    "capacity_expansion",
    "merger_acquisition",
    "share_buyback",
    "shareholder_increase",
    "shareholder_decrease",
    "regulatory_penalty",
    "lawsuit_risk",
    "financing",
    "management_change",
    "supply_chain",
    "price_change",
    "rumor",
    "unknown",
]

SourceType = Literal[
    "exchange_disclosure",
    "company_announcement",
    "official_policy",
    "financial_media",
    "data_platform",
    "research_summary",
    "company_website",
    "social_media",
    "forum",
    "short_video",
    "unknown",
]

RiskType = Literal[
    "already_priced_in",
    "overheated",
    "rumor_only",
    "weak_source",
    "conflicting_info",
    "one_time_gain",
    "valuation_pressure",
    "liquidity_risk",
    "policy_uncertainty",
    "earnings_uncertainty",
    "negative_event",
    "unknown",
]

RiskLevel = Literal["low", "medium", "high", "critical"]

QuantFocusType = Literal[
    "check_trend",
    "check_volume_confirmation",
    "check_overheat",
    "check_volatility",
    "check_drawdown",
    "check_relative_strength",
    "check_liquidity",
    "check_gap_risk",
    "check_breakout",
    "check_breakdown",
]


class RunContext(BaseModel):
    """
    一次 AI 搜索任务的整体背景。
    """

    model_config = ConfigDict(extra="forbid")

    as_of_time: datetime = Field(..., description="AI 生成这个候选池的时间点")
    market: str = Field(..., description="市场，例如 A股")
    search_scope: str = Field(..., description="搜索范围，例如 全市场 / 新能源 / 半导体")
    search_goal: str = Field(..., description="搜索目标")
    ai_model: str = Field(..., description="使用的 AI 模型")
    language: str = Field(default="zh-CN")
    notes: str | None = None


class SignalSource(BaseModel):
    """
    AI 使用的信息来源。
    """

    model_config = ConfigDict(extra="forbid")

    source_id: str
    source_type: SourceType
    source_name: str | None = None
    title: str
    url: str | None = None
    publish_time: datetime | None = None
    crawl_time: datetime | None = None

    source_tier: int = Field(..., ge=1, le=5, description="1最高可信，5最低可信")
    source_confidence: float = Field(..., ge=0, le=1)

    is_official: bool = False
    is_duplicate: bool = False

    content_summary: str | None = None
    key_quote: str | None = None


class SignalEvent(BaseModel):
    """
    单个事件。
    """

    model_config = ConfigDict(extra="forbid")

    event_id: str
    event_type: EventType
    event_time: datetime | None = None
    publish_time: datetime | None = None

    title: str
    summary: str
    evidence_excerpt: str | None = None

    sentiment_score: float = Field(..., ge=-1, le=1)
    event_strength: float = Field(..., ge=0, le=1)
    certainty: float = Field(..., ge=0, le=1)
    freshness_score: float = Field(..., ge=0, le=1)
    directness_score: float = Field(..., ge=0, le=1)

    source_ids: list[str] = Field(..., min_length=1)

    positive_points: list[str] = Field(default_factory=list)
    negative_points: list[str] = Field(default_factory=list)


class SignalRiskFlag(BaseModel):
    """
    结构化风险标记。
    """

    model_config = ConfigDict(extra="forbid")

    risk_type: RiskType
    risk_level: RiskLevel
    description: str
    related_event_ids: list[str] = Field(default_factory=list)


class SignalContradiction(BaseModel):
    """
    信息冲突。
    """

    model_config = ConfigDict(extra="forbid")

    contradiction_type: str
    description: str
    source_ids: list[str] = Field(default_factory=list)
    impact_on_confidence: float = Field(default=0.0, ge=-1, le=1)


class SignalQuantFocus(BaseModel):
    """
    AI 提醒量化模型重点检查什么。
    """

    model_config = ConfigDict(extra="forbid")

    focus_type: QuantFocusType
    reason: str


class SignalCandidate(BaseModel):
    """
    一只候选股票。
    """

    model_config = ConfigDict(extra="forbid")

    stock_code: str = Field(..., description="内部统一股票代码，例如 300750.SZ")
    stock_name: str
    market: str = Field(..., description="A股 / 港股 / 美股")
    exchange: str = Field(..., description="SZ / SH / BJ 等")

    industry: str | None = None
    sector: str | None = None
    concept_tags: list[str] = Field(default_factory=list)

    candidate_reason: str
    ai_overall_score: float = Field(..., ge=0, le=1)
    ai_confidence: float = Field(..., ge=0, le=1)
    expected_horizon_days: list[int] = Field(default_factory=lambda: [3, 5, 10])

    events: list[SignalEvent] = Field(..., min_length=1)
    sources: list[SignalSource] = Field(..., min_length=1)
    risk_flags: list[SignalRiskFlag] = Field(default_factory=list)
    contradictions: list[SignalContradiction] = Field(default_factory=list)
    quant_focus: list[SignalQuantFocus] = Field(..., min_length=1)

    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("stock_code")
    @classmethod
    def validate_stock_code(cls, value: str) -> str:
        value = value.strip().upper()

        if not re.fullmatch(r"\d{6}\.(SZ|SH|BJ)", value):
            raise ValueError(
                "stock_code 必须使用 000001.SZ / 600000.SH / 430001.BJ 这种格式"
            )

        return value

    @model_validator(mode="after")
    def validate_references(self) -> "SignalCandidate":
        source_ids = {source.source_id for source in self.sources}
        event_ids = {event.event_id for event in self.events}

        for event in self.events:
            for source_id in event.source_ids:
                if source_id not in source_ids:
                    raise ValueError(
                        f"event {event.event_id} 引用了不存在的 source_id: {source_id}"
                    )

        for risk in self.risk_flags:
            for event_id in risk.related_event_ids:
                if event_id not in event_ids:
                    raise ValueError(
                        f"risk {risk.risk_type} 引用了不存在的 event_id: {event_id}"
                    )

        for contradiction in self.contradictions:
            for source_id in contradiction.source_ids:
                if source_id not in source_ids:
                    raise ValueError(
                        f"contradiction 引用了不存在的 source_id: {source_id}"
                    )

        return self


class StockLensSignalPackage(BaseModel):
    """
    StockLens Signal Package v1.0

    这是 AI 网页信息传入 StockLens AI 的固定格式。
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"]
    run_context: RunContext
    candidates: list[SignalCandidate] = Field(..., min_length=1, max_length=50)