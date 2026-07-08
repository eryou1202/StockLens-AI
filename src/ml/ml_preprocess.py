from __future__ import annotations

import pandas as pd


class MLPreprocessor:
    """Select safe numeric research features and create sklearn pipelines."""

    EXCLUDED_COLUMNS = {
        "sample_id", "symbol", "stock_name", "as_of_date", "price_time", "current_price",
        "source", "sample_interval_days", "lookback_days", "label_status", "label_error",
    }
    EXCLUDED_PREFIXES = (
        "future_return_",
        "future_excess_return_",
        "future_rank_pct_",
        "future_top30_",
        "future_bottom30_",
        "hit_",
        "future_max_drawdown_",
    )

    def select_features(self, train_frame: pd.DataFrame, target: str) -> list[str]:
        columns: list[str] = []
        for column in train_frame.columns:
            if column == target or column in self.EXCLUDED_COLUMNS:
                continue
            if column.startswith(self.EXCLUDED_PREFIXES):
                continue
            numeric = pd.to_numeric(train_frame[column], errors="coerce")
            if numeric.notna().any():
                columns.append(column)
        return columns

    @staticmethod
    def numeric_frame(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
        return frame[features].apply(pd.to_numeric, errors="coerce")

    @staticmethod
    def build_pipeline(model_type: str):
        try:
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.impute import SimpleImputer
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import StandardScaler
        except ImportError as exc:
            raise ImportError("需要安装 scikit-learn 才能训练研究模型") from exc

        if model_type == "logistic":
            return Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)),
            ])
        if model_type == "random_forest_regressor":
            return Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("model", RandomForestRegressor(
                    n_estimators=300,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1,
                )),
            ])
        raise ValueError(f"unsupported model type: {model_type}")
