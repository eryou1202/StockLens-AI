from __future__ import annotations

from src.recommendation.recommendation_explainer import RecommendationExplainer
from scripts.run_recommendations import run_recommendation_analysis


def main() -> None:
    try:
        recommendations = run_recommendation_analysis()
    except Exception as exc:
        print(f"推荐报告生成失败：{type(exc).__name__}: {exc}")
        return

    print("StockLens Recommendation Report")
    print("StockLens AI 仅用于研究和辅助决策，不构成投资建议。")
    for index, recommendation in enumerate(recommendations):
        if index:
            print("\n" + "=" * 72 + "\n")
        print(RecommendationExplainer.format_report(recommendation))


if __name__ == "__main__":
    main()
