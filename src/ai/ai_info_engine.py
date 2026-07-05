from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timedelta

from src.models.schemas import AICandidate, FeedbackSummary


class AIInfoEngine(ABC):
    """
    AI 信息引擎接口。

    这里只负责输出候选股票池。
    """

    @abstractmethod
    def generate_candidates(self, as_of_time: datetime) -> list[AICandidate]:
        raise NotImplementedError

    def receive_feedback(self, summary: FeedbackSummary) -> None:
        """
        TODO:
        - AI 根据反馈调整搜索重点
        - 不要让 AI 直接改量化参数
        """
        return None


class MockAIInfoEngine(AIInfoEngine):
    def generate_candidates(self, as_of_time: datetime) -> list[AICandidate]:
        return [
            AICandidate(
                stock_code="000001.SZ",
                stock_name="平安银行",
                as_of_time=as_of_time,
                event_time=as_of_time - timedelta(hours=1),
                event_type="行业利好",
                event_summary="Mock：AI 检测到银行板块存在正面政策讨论。",
                sentiment_score=0.45,
                event_strength=0.55,
                source_confidence=0.70,
                ai_confidence=0.60,
                risk_flags=["政策影响需要进一步验证", "消息可能已被市场部分反映"],
                source_urls=["https://example.com/news/000001"],
            ),
            AICandidate(
                stock_code="300750.SZ",
                stock_name="宁德时代",
                as_of_time=as_of_time,
                event_time=as_of_time - timedelta(hours=2),
                event_type="产业链利好",
                event_summary="Mock：AI 检测到新能源产业链相关消息热度上升。",
                sentiment_score=0.68,
                event_strength=0.75,
                source_confidence=0.65,
                ai_confidence=0.62,
                risk_flags=["短期热度较高，需检查是否追高"],
                source_urls=["https://example.com/news/300750"],
            ),
        ]
