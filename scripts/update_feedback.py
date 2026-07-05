from __future__ import annotations

from src.config.settings import load_settings
from src.data.provider_factory import create_market_data_provider
from src.feedback.feedback_engine import FeedbackEngine


def main() -> None:
    settings = load_settings()
    print("StockLens Feedback Update\n")
    print(f"database: {settings.database_path}")
    print(f"provider: {settings.market_provider}\n")

    try:
        # 反馈需要持续看到新增交易日，避免历史 CSV 缓存遮蔽未来行情。
        provider = create_market_data_provider(
            name=settings.market_provider,
            cache_dir=settings.cache_dir,
            use_cache=False,
        )
        engine = FeedbackEngine(
            database_path=settings.database_path,
            market_data_provider=provider,
            lookahead_days=30,
            adjust_type=settings.market_adjust_type,
        )
        summary = engine.update_all_pending()
    except Exception as exc:
        print(f"反馈更新未完成：{type(exc).__name__}: {exc}")
        return

    for key in ("signals_found", "updated", "pending", "partial", "complete", "failed"):
        print(f"{key}: {summary[key]}")
    if summary["signals_found"] == 0:
        print("\n当前没有待回填的信号。")


if __name__ == "__main__":
    main()
