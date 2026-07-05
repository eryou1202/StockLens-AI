from __future__ import annotations

import argparse
from datetime import datetime, timedelta

from src.config.settings import load_settings
from src.data.provider_factory import create_market_data_provider
from src.quant.explanation_builder import ExplanationBuilder
from src.quant.feature_builder import QuantFeatureBuilder
from src.quant.rule_scorer import RuleScorer


def _format(value: float | None) -> str:
    return "None" if value is None else f"{value:.6f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Print StockLens quant features and rule scores.")
    parser.add_argument("--symbol", default="300750.SZ")
    parser.add_argument("--lookback-days", type=int, default=180)
    args = parser.parse_args()

    settings = load_settings()
    provider = create_market_data_provider(
        name=settings.market_provider,
        cache_dir=settings.cache_dir,
        use_cache=True,
    )

    end_time = datetime.now()
    start_time = end_time - timedelta(days=max(120, min(args.lookback_days, 180)))
    market_data = provider.get_bars(
        symbol=args.symbol,
        start_time=start_time,
        end_time=end_time,
        frequency=settings.market_frequency,
        adjust_type=settings.market_adjust_type,
    )

    features = QuantFeatureBuilder().build(market_data)
    scores = RuleScorer().score(features)
    reason, reason_tags = ExplanationBuilder().build(features, scores)

    print("StockLens Quant Feature Test\n")
    print(f"symbol: {features.symbol}")
    print(f"provider: {features.provider}")
    print(f"rows: {features.data_rows}")
    print(f"data_quality_score: {features.data_quality_score:.1f}")
    print(f"warnings: {', '.join(features.warnings) if features.warnings else '-'}")

    print("\nPrices:")
    print(f"latest_close: {_format(features.latest_close)}")

    print("\nReturns:")
    print(f"return_1d: {_format(features.return_1d)}")
    print(f"return_5d: {_format(features.return_5d)}")
    print(f"return_20d: {_format(features.return_20d)}")

    print("\nMoving Averages:")
    print(f"ma5: {_format(features.ma5)}")
    print(f"ma20: {_format(features.ma20)}")
    print(f"ma60: {_format(features.ma60)}")
    print(f"ma5_ma20_gap: {_format(features.ma5_ma20_gap)}")
    print(f"ma20_ma60_gap: {_format(features.ma20_ma60_gap)}")

    print("\nVolume:")
    print(f"volume_ratio_5d: {_format(features.volume_ratio_5d)}")
    print(f"amount_ratio_5d: {_format(features.amount_ratio_5d)}")

    print("\nRisk:")
    print(f"volatility_20d: {_format(features.volatility_20d)}")
    print(f"max_drawdown_20d: {_format(features.max_drawdown_20d)}")
    print(f"atr_14: {_format(features.atr_14)}")

    print("\nIndicators:")
    print(f"rsi_14: {_format(features.rsi_14)}")
    print(f"macd_hist: {_format(features.macd_hist)}")
    print(f"bollinger_position: {_format(features.bollinger_position)}")

    print("\nScores:")
    print(f"trend_score: {scores.trend_score:.2f}")
    print(f"momentum_score: {scores.momentum_score:.2f}")
    print(f"volume_score: {scores.volume_score:.2f}")
    print(f"risk_score: {scores.risk_score:.2f}")
    print(f"overheat_score: {scores.overheat_score:.2f}")
    print(f"quant_score: {scores.quant_score:.2f}")
    print(f"heuristic_prob_up_5d: {scores.heuristic_prob_up_5d:.4f}")
    print(f"quant_decision: {scores.quant_decision}")

    print("\nReason:")
    print(reason)
    print(f"tags: {', '.join(reason_tags)}")


if __name__ == "__main__":
    main()
