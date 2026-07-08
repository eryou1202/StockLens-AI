from __future__ import annotations

import pandas as pd


RELATIVE_TARGET_HORIZONS = (1, 2, 3, 5, 10, 20)
RELATIVE_LABEL_PREFIXES = (
    "future_excess_return_",
    "future_rank_pct_",
    "future_top30_",
    "future_bottom30_",
)


class RelativeTargetLabelBuilder:
    """Build future-only cross-sectional labels after all samples are assembled."""

    @staticmethod
    def apply(frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.copy()
        if "as_of_date" not in result.columns:
            return result
        dates = result["as_of_date"]
        for horizon in RELATIVE_TARGET_HORIZONS:
            return_column = f"future_return_{horizon}d"
            if return_column not in result.columns:
                continue
            values = pd.to_numeric(result[return_column], errors="coerce")
            grouped = values.groupby(dates, dropna=False)
            median = grouped.transform("median")
            ordinal_rank = grouped.rank(method="average")
            group_count = grouped.transform("count")
            rank_pct = (ordinal_rank - 1.0) / (group_count - 1.0)
            rank_pct = rank_pct.where(group_count > 1, 0.5).where(values.notna())

            result[f"future_excess_return_{horizon}d"] = values - median
            result[f"future_rank_pct_{horizon}d"] = rank_pct
            result[f"future_top30_{horizon}d"] = RelativeTargetLabelBuilder._binary_label(
                rank_pct, "top"
            )
            result[f"future_bottom30_{horizon}d"] = RelativeTargetLabelBuilder._binary_label(
                rank_pct, "bottom"
            )
        return result

    @staticmethod
    def _binary_label(rank_pct: pd.Series, side: str) -> pd.Series:
        label = pd.Series(pd.NA, index=rank_pct.index, dtype="Int64")
        valid = rank_pct.notna()
        if side == "top":
            label.loc[valid] = rank_pct.loc[valid].ge(0.70).astype(int)
        else:
            label.loc[valid] = rank_pct.loc[valid].le(0.30).astype(int)
        return label
