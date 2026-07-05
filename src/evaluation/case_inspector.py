from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


class CaseInspector:
    IDENTITY_COLUMNS = [
        "symbol", "stock_name", "as_of_time", "final_level", "final_score",
        "ai_view", "quant_view", "quant_decision", "quant_score",
        "heuristic_prob_up_5d",
    ]
    OUTCOME_COLUMNS = [
        "future_return_1d", "future_return_3d", "future_return_5d",
        "future_return_10d", "future_max_drawdown_5d",
    ]
    FEATURE_COLUMNS = [
        "return_5d", "return_20d", "ma5_ma20_gap", "ma20_ma60_gap",
        "close_ma20_gap", "volume_ratio_5d", "amount_ratio_5d",
        "volatility_20d", "max_drawdown_20d", "atr_14", "rsi_14",
        "macd_hist", "bollinger_position", "trend_score", "momentum_score",
        "volume_score", "risk_score", "overheat_score",
    ]
    CASE_COLUMNS = IDENTITY_COLUMNS + OUTCOME_COLUMNS + FEATURE_COLUMNS
    NUMERIC_COLUMNS = [
        "final_score", "quant_score", "heuristic_prob_up_5d",
        *OUTCOME_COLUMNS,
        *FEATURE_COLUMNS,
    ]

    def __init__(self, dataset_path: str = "data/ml_dataset.csv"):
        self.dataset_path = Path(dataset_path)

    def load_complete(self) -> pd.DataFrame:
        if not self.dataset_path.exists():
            raise FileNotFoundError(str(self.dataset_path))
        frame = pd.read_csv(self.dataset_path)
        if "feedback_status" not in frame:
            return frame.iloc[0:0].copy()
        status = frame["feedback_status"].astype("string").str.strip().str.lower()
        complete = frame.loc[status == "complete"].copy()
        for column in self.NUMERIC_COLUMNS:
            if column in complete:
                complete[column] = pd.to_numeric(complete[column], errors="coerce")
        if "quant_decision" in complete:
            complete["quant_decision"] = (
                complete["quant_decision"].astype("string").str.strip().str.lower()
            )
        return complete

    def top_bad_support_cases(self, n: int = 20) -> pd.DataFrame:
        frame = self.load_complete()
        if "quant_decision" not in frame or "future_return_5d" not in frame:
            return self._select_columns(frame.iloc[0:0])
        cases = frame.loc[
            (frame["quant_decision"] == "support")
            & frame["future_return_5d"].notna()
            & (frame["future_return_5d"] <= 0)
        ].sort_values("future_return_5d", ascending=True, kind="stable")
        return self._select_columns(cases.head(max(0, int(n))))

    def top_missed_reject_cases(self, n: int = 20) -> pd.DataFrame:
        frame = self.load_complete()
        if "quant_decision" not in frame or "future_return_5d" not in frame:
            return self._select_columns(frame.iloc[0:0])
        cases = frame.loc[
            (frame["quant_decision"] == "reject")
            & frame["future_return_5d"].notna()
            & (frame["future_return_5d"] > 0)
        ].sort_values("future_return_5d", ascending=False, kind="stable")
        return self._select_columns(cases.head(max(0, int(n))))

    def extreme_score_mismatch_cases(self, n: int = 20) -> pd.DataFrame:
        frame = self.load_complete()
        if "quant_score" not in frame or "future_return_5d" not in frame:
            empty = self._select_columns(frame.iloc[0:0])
            empty["mismatch_score"] = pd.Series(dtype="float64")
            return empty

        cases = frame.loc[
            frame["quant_score"].notna() & frame["future_return_5d"].notna()
        ].copy()
        absolute_return = cases["future_return_5d"].abs()
        cases["mismatch_score"] = cases["quant_score"] * absolute_return
        positive = cases["future_return_5d"] > 0
        cases.loc[positive, "mismatch_score"] = (
            (100.0 - cases.loc[positive, "quant_score"])
            * absolute_return.loc[positive]
        )
        cases = cases.sort_values("mismatch_score", ascending=False, kind="stable")
        selected = self._select_columns(cases.head(max(0, int(n))))
        selected["mismatch_score"] = cases.head(max(0, int(n)))["mismatch_score"].to_numpy()
        return selected

    def print_report(self, n: int = 10) -> None:
        complete = self.load_complete()
        print("StockLens Case Inspector v1.0\n")
        print(f"complete samples: {len(complete)}")
        if complete.empty:
            print("没有 complete 样本，暂时无法检查错误案例。")
            print("\n当前历史样本来自测试型 historical batch，不代表真实历史回测表现。")
            return

        self._print_cases("Bad support cases", self.top_bad_support_cases(n))
        self._print_cases("Missed reject cases", self.top_missed_reject_cases(n))
        self._print_cases("Extreme score mismatch cases", self.extreme_score_mismatch_cases(n))
        print("\n当前历史样本来自测试型 historical batch，不代表真实历史回测表现。")
        print("Case Inspector 只用于开发和诊断规则，不用于真实交易结论。")

    def _select_columns(self, frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.copy()
        for column in self.CASE_COLUMNS:
            if column not in result:
                result[column] = pd.NA
        return result[self.CASE_COLUMNS].reset_index(drop=True)

    @classmethod
    def _print_cases(cls, title: str, cases: pd.DataFrame) -> None:
        print(f"\n{title} ({len(cases)}):")
        if cases.empty:
            print("  none")
            return
        for index, row in enumerate(cases.to_dict(orient="records"), start=1):
            mismatch = row.get("mismatch_score")
            mismatch_text = "" if mismatch is None or pd.isna(mismatch) else f" mismatch={mismatch:.4f}"
            print(
                f"  [{index}] {row.get('symbol', '-')} {cls._text(row.get('stock_name'))} "
                f"{cls._text(row.get('as_of_time'))} | final={cls._text(row.get('final_level'))} "
                f"decision={cls._text(row.get('quant_decision'))} q={cls._number(row.get('quant_score'))} "
                f"prob={cls._percent(row.get('heuristic_prob_up_5d'))} "
                f"future5={cls._percent(row.get('future_return_5d'))} "
                f"dd5={cls._percent(row.get('future_max_drawdown_5d'))}{mismatch_text}"
            )
            print(
                "      "
                f"ret5={cls._percent(row.get('return_5d'))} "
                f"ret20={cls._percent(row.get('return_20d'))} "
                f"ma5/20={cls._percent(row.get('ma5_ma20_gap'))} "
                f"ma20/60={cls._percent(row.get('ma20_ma60_gap'))} "
                f"vol_ratio={cls._number(row.get('volume_ratio_5d'))} "
                f"rsi={cls._number(row.get('rsi_14'))} "
                f"trend={cls._number(row.get('trend_score'))} "
                f"momentum={cls._number(row.get('momentum_score'))} "
                f"risk={cls._number(row.get('risk_score'))} "
                f"overheat={cls._number(row.get('overheat_score'))}"
            )

    @staticmethod
    def _text(value: Any) -> str:
        return "-" if value is None or pd.isna(value) else str(value)

    @staticmethod
    def _number(value: Any) -> str:
        return "-" if value is None or pd.isna(value) else f"{float(value):.2f}"

    @staticmethod
    def _percent(value: Any) -> str:
        return "-" if value is None or pd.isna(value) else f"{float(value):.2%}"
