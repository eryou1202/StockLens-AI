from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(float("nan"), index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce")


def _text(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(None, index=frame.index, dtype="object")
    return frame[column].astype("string").str.strip().str.lower()


def _hits(frame: pd.DataFrame) -> pd.Series:
    if "hit_5d" not in frame:
        return pd.Series(float("nan"), index=frame.index, dtype="float64")
    mapping = {
        "true": 1.0, "1": 1.0, "1.0": 1.0, "yes": 1.0,
        "false": 0.0, "0": 0.0, "0.0": 0.0, "no": 0.0,
    }
    return frame["hit_5d"].astype("string").str.strip().str.lower().map(mapping).astype("float64")


def _mean(series: pd.Series) -> float | None:
    clean = series.dropna()
    return None if clean.empty else float(clean.mean())


def _stats(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "count": int(len(frame)),
        "hit_rate_5d": _mean(_hits(frame)),
        "avg_future_return_5d": _mean(_numeric(frame, "future_return_5d")),
        "avg_drawdown_5d": _mean(_numeric(frame, "future_max_drawdown_5d")),
    }


def _format(value: float | None) -> str:
    return "-" if value is None or pd.isna(value) else f"{value:.2%}"


def _print_groups(
    title: str,
    frame: pd.DataFrame,
    column: str,
    expected: list[str],
) -> dict[str, dict[str, Any]]:
    values = _text(frame, column)
    groups = {name: _stats(frame.loc[values == name]) for name in expected}
    print(f"\n{title}:")
    for name, stats in groups.items():
        print(
            f"  {name}: count={stats['count']}, "
            f"hit_rate_5d={_format(stats['hit_rate_5d'])}, "
            f"avg_return_5d={_format(stats['avg_future_return_5d'])}, "
            f"avg_drawdown_5d={_format(stats['avg_drawdown_5d'])}"
        )
    return groups


def _print_symbol_decision_groups(frame: pd.DataFrame) -> None:
    symbols = (
        frame["symbol"].astype("string").str.strip()
        if "symbol" in frame
        else pd.Series(None, index=frame.index, dtype="object")
    )
    decisions = _text(frame, "quant_decision")
    print("\nBy symbol + quant_decision:")
    printed = False
    for symbol in sorted(str(value) for value in symbols.dropna().unique()):
        for decision in ("support", "uncertain", "reject"):
            group = frame.loc[(symbols == symbol) & (decisions == decision)]
            if group.empty:
                continue
            stats = _stats(group)
            printed = True
            print(
                f"  symbol={symbol}, quant_decision={decision}, count={stats['count']}, "
                f"hit_rate_5d={_format(stats['hit_rate_5d'])}, "
                f"avg_return_5d={_format(stats['avg_future_return_5d'])}"
            )
    if not printed:
        print("  no grouped cases")


def main() -> None:
    dataset_path = Path("data/ml_dataset.csv")
    print("StockLens Rule Baseline Evaluation\n")
    if not dataset_path.exists():
        print("data/ml_dataset.csv 不存在，请先运行：py -m scripts.build_ml_dataset")
        return

    try:
        frame = pd.read_csv(dataset_path)
    except Exception as exc:
        print(f"数据集读取失败：{type(exc).__name__}: {exc}")
        return

    status = _text(frame, "feedback_status")
    complete = frame.loc[status == "complete"].copy()
    print(f"complete samples: {len(complete)}")
    if complete.empty:
        print("没有 complete 样本，暂时无法评估规则 baseline。")
        print("\n当前结果来自测试型历史样本，不代表真实历史回测表现。")
        return

    decision_groups = _print_groups(
        "By quant_decision",
        complete,
        "quant_decision",
        ["support", "uncertain", "reject"],
    )
    _print_symbol_decision_groups(complete)
    _print_groups(
        "By final_level",
        complete,
        "final_level",
        ["strong_watch", "watch", "risky", "avoid"],
    )

    scores = _numeric(complete, "quant_score")
    returns = _numeric(complete, "future_return_5d")
    paired = pd.DataFrame({"score": scores, "return": returns}).dropna()
    correlation = None
    if len(paired) >= 3 and paired["score"].nunique() > 1 and paired["return"].nunique() > 1:
        correlation = float(paired["score"].corr(paired["return"], method="pearson"))
    print("\nquant_score vs future_return_5d:")
    print(f"  paired_samples: {len(paired)}")
    print(f"  pearson_correlation: {'-' if correlation is None else f'{correlation:.6f}'}")

    probabilities = _numeric(complete, "heuristic_prob_up_5d")
    hit_values = _hits(complete)
    labels = ["0.0-0.4", "0.4-0.5", "0.5-0.6", "0.6-0.7", "0.7-1.0"]
    buckets = pd.cut(
        probabilities,
        bins=[0.0, 0.4, 0.5, 0.6, 0.7, 1.0000001],
        labels=labels,
        include_lowest=True,
        right=False,
    )
    print("\nheuristic_prob_up_5d calibration:")
    for label in labels:
        mask = buckets == label
        print(
            f"  {label}: count={int(mask.sum())}, "
            f"avg_predicted_prob={_format(_mean(probabilities.loc[mask]))}, "
            f"actual_hit_rate={_format(_mean(hit_values.loc[mask]))}"
        )

    support_rate = decision_groups["support"]["hit_rate_5d"]
    reject_rate = decision_groups["reject"]["hit_rate_5d"]
    if support_rate is not None and reject_rate is not None and support_rate < reject_rate:
        print(
            "\n警告：当前规则 baseline 的 support/reject 排序可能无效，"
            "请不要据此做真实交易判断。"
        )
    if len(complete) < 30:
        print("\n完整样本少于 30，当前分组统计稳定性较低。")
    print("\n当前结果来自测试型历史样本，不代表真实历史回测表现。")


if __name__ == "__main__":
    main()
