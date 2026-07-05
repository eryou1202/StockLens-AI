from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime

from src.models.schemas import (
    AICandidate,
    DecisionLevel,
    MarketDataBundle,
    QuantFeatures,
    QuantResult,
)
from src.quant.explanation_builder import ExplanationBuilder
from src.quant.feature_builder import QuantFeatureBuilder
from src.quant.feature_schema import QuantFeatureVector, QuantScoreBreakdown
from src.quant.rule_scorer import RuleScorer, clip


class QuantEngine(ABC):
    """量化曲线分析接口。"""

    @abstractmethod
    def analyze(
        self,
        candidate: AICandidate,
        market_data: MarketDataBundle,
        as_of_time: datetime,
        index_data: MarketDataBundle | None = None,
        industry_data: MarketDataBundle | None = None,
    ) -> QuantResult:
        raise NotImplementedError


class RuleBasedQuantEngine(QuantEngine):
    """StockLens Rule-Based Quant Baseline v1.0。"""

    model_version = "rule_v1"

    def __init__(
        self,
        feature_builder: QuantFeatureBuilder | None = None,
        scorer: RuleScorer | None = None,
        explanation_builder: ExplanationBuilder | None = None,
    ) -> None:
        self.feature_builder = feature_builder or QuantFeatureBuilder()
        self.scorer = scorer or RuleScorer()
        self.explanation_builder = explanation_builder or ExplanationBuilder()

    def analyze(
        self,
        candidate: AICandidate,
        market_data: MarketDataBundle,
        as_of_time: datetime,
        index_data: MarketDataBundle | None = None,
        industry_data: MarketDataBundle | None = None,
    ) -> QuantResult:
        del index_data, industry_data  # v1 暂不使用指数和行业数据，保留参数以兼容 pipeline。

        features = self.feature_builder.build(market_data)
        scores = self.scorer.score(features)
        quant_reason, reason_tags = self.explanation_builder.build(features, scores)

        return QuantResult(
            symbol=candidate.stock_code,
            as_of_time=as_of_time,
            trend_state=self._trend_state(scores),
            volume_state=self._volume_state(features, scores),
            volatility_state=self._volatility_state(scores),
            momentum_score=scores.momentum_score,
            trend_score=scores.trend_score,
            risk_score=scores.risk_score,
            overheat_score=scores.overheat_score,
            prob_up_5d=scores.heuristic_prob_up_5d,
            prob_outperform_index_5d=clip(scores.heuristic_prob_up_5d - 0.05, 0.05, 0.85),
            predicted_return_5d=(scores.quant_score - 50.0) / 1000.0,
            max_drawdown_risk_5d=min(0.5, scores.risk_score / 200.0),
            model_confidence=clip(features.data_quality_score / 100.0, 0.2, 1.0),
            quant_decision=self._decision_level(scores),
            quant_reason=quant_reason.split("；"),
            features=self._legacy_features(
                symbol=candidate.stock_code,
                as_of_time=as_of_time,
                features=features,
                scores=scores,
                reason_tags=reason_tags,
            ),
            model_version=self.model_version,
        )

    @staticmethod
    def _trend_state(scores: QuantScoreBreakdown) -> str:
        if scores.trend_score >= 70:
            return "uptrend"
        if scores.trend_score <= 40:
            return "downtrend"
        return "sideways"

    @staticmethod
    def _volume_state(features: QuantFeatureVector, scores: QuantScoreBreakdown) -> str:
        ratio = features.volume_ratio_5d
        if ratio is not None and ratio > 4:
            return "abnormal_expansion"
        if scores.volume_score >= 70:
            return "moderate_expansion"
        if ratio is None:
            return "unknown"
        if ratio < 0.8:
            return "shrinking"
        return "normal"

    @staticmethod
    def _volatility_state(scores: QuantScoreBreakdown) -> str:
        if scores.risk_score >= 75:
            return "high"
        if scores.risk_score <= 40:
            return "low"
        return "medium"

    @staticmethod
    def _decision_level(scores: QuantScoreBreakdown) -> DecisionLevel:
        if scores.quant_decision == "support":
            return DecisionLevel.WATCH
        if scores.quant_decision == "uncertain":
            return DecisionLevel.RISKY
        return DecisionLevel.AVOID

    @staticmethod
    def _legacy_features(
        symbol: str,
        as_of_time: datetime,
        features: QuantFeatureVector,
        scores: QuantScoreBreakdown,
        reason_tags: list[str],
    ) -> QuantFeatures:
        return QuantFeatures(
            symbol=symbol,
            as_of_time=as_of_time,
            return_1d=features.return_1d,
            return_5d=features.return_5d,
            return_20d=features.return_20d,
            ma5=features.ma5,
            ma20=features.ma20,
            ma60=features.ma60,
            ma5_above_ma20=(
                features.ma5 > features.ma20
                if features.ma5 is not None and features.ma20 is not None
                else None
            ),
            ma20_above_ma60=(
                features.ma20 > features.ma60
                if features.ma20 is not None and features.ma60 is not None
                else None
            ),
            volume_ratio_5d=features.volume_ratio_5d,
            volatility_20d=features.volatility_20d,
            max_drawdown_20d=features.max_drawdown_20d,
            overheat_score=scores.overheat_score / 100.0,
            liquidity_score=scores.volume_score / 100.0,
            extra={
                "feature_vector": features.model_dump(mode="json"),
                "score_breakdown": scores.model_dump(mode="json"),
                "reason_tags": reason_tags,
                "internal_quant_decision": scores.quant_decision,
            },
        )
