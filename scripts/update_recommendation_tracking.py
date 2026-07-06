from __future__ import annotations

from src.config.settings import load_settings
from src.data.provider_factory import create_market_data_provider
from src.tracking.recommendation_tracker import RecommendationTracker


def main() -> None:
    settings = load_settings()
    provider = create_market_data_provider(settings.market_provider, settings.cache_dir, True)
    try:
        summary = RecommendationTracker(
            settings.database_path, provider, settings.market_adjust_type
        ).update_future_performance()
    except Exception as exc:
        print(f"更新追踪表现失败：{type(exc).__name__}: {exc}")
        return
    print("StockLens Recommendation Tracking Update")
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
