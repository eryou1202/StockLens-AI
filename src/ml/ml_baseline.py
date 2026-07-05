from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import pandas as pd


class MLBaselineTrainer:
    FEATURE_COLUMNS = [
        "ai_sentiment_score", "ai_event_strength", "ai_source_confidence", "ai_confidence",
        "ai_overall_score",
        "return_1d", "return_3d", "return_5d", "return_10d", "return_20d", "return_60d",
        "ma5_ma20_gap", "ma20_ma60_gap", "close_ma20_gap",
        "volume_ratio_5d", "volume_ratio_20d", "amount_ratio_5d", "amount_ratio_20d",
        "volatility_20d", "max_drawdown_20d", "atr_14", "rsi_14", "macd_hist",
        "bollinger_position", "trend_score", "momentum_score", "volume_score", "risk_score",
        "overheat_score", "macd_score", "quant_score", "heuristic_prob_up_5d",
    ]

    def __init__(self, dataset_path: str = "data/ml_dataset.csv"):
        self.dataset_path = Path(dataset_path)

    def train(self, target: str = "hit_5d") -> dict[str, Any]:
        if not self.dataset_path.exists():
            return {
                "status": "dataset_missing",
                "message": f"数据集不存在：{self.dataset_path}",
                "feature_columns": [],
            }

        frame = pd.read_csv(self.dataset_path)
        if target not in frame.columns:
            return {
                "status": "target_missing",
                "message": f"数据集中缺少目标列：{target}",
                "feature_columns": [],
            }

        status = frame.get("feedback_status", pd.Series("pending", index=frame.index))
        status = status.astype("string").str.strip().str.lower()
        target_values = self._binary_target(frame[target])
        usable = frame.loc[(status == "complete") & target_values.notna()].copy()
        usable[target] = target_values.loc[usable.index].astype(int)
        feature_columns = [column for column in self.FEATURE_COLUMNS if column in usable.columns]

        usable["_sort_time"] = pd.to_datetime(usable.get("as_of_time"), errors="coerce")
        usable = usable.sort_values("_sort_time", kind="stable").reset_index(drop=True)
        labels = usable[target].astype(int)
        train_size = max(1, min(len(usable) - 1, int(len(usable) * 0.70))) if len(usable) >= 2 else len(usable)
        y_train = labels.iloc[:train_size]
        y_test = labels.iloc[train_size:]

        diagnostics: dict[str, Any] = {
            "complete_samples": int(len(usable)),
            "train_samples": int(len(y_train)),
            "test_samples": int(len(y_test)),
            "feature_columns": feature_columns,
            "label_distribution_total": self._distribution(labels),
            "label_distribution_train": self._distribution(y_train),
            "label_distribution_test": self._distribution(y_test),
            "positive_rate_test": self._mean_or_none(y_test),
            "baseline_accuracy": self._majority_accuracy(y_test),
        }

        if len(usable) < 30:
            return {
                "status": "insufficient_samples",
                "message": "完整反馈样本少于 30，暂不训练正式模型。当前只能检查数据流程。",
                **diagnostics,
            }
        if not feature_columns:
            return {
                "status": "no_features",
                "message": "没有可用于训练的数值特征列。",
                **diagnostics,
            }
        if labels.nunique() < 2:
            return {
                "status": "insufficient_classes",
                "message": "完整反馈标签只有一个类别，暂时无法训练二分类模型。",
                **diagnostics,
            }
        if y_train.nunique() < 2:
            return {
                "status": "insufficient_train_classes",
                "message": "按时间切分后的训练集只有一个类别，暂时无法训练。",
                **diagnostics,
            }

        try:
            from sklearn.impute import SimpleImputer
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import (
                accuracy_score,
                confusion_matrix,
                precision_score,
                recall_score,
                roc_auc_score,
            )
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            return {
                "status": "dependency_missing",
                "message": "未安装 scikit-learn，请运行：pip install scikit-learn",
                **diagnostics,
            }

        features = usable[feature_columns].apply(pd.to_numeric, errors="coerce")
        for column in feature_columns:
            if features[column].isna().all():
                features[column] = 0.0
        x_train, x_test = features.iloc[:train_size], features.iloc[train_size:]

        model = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
            ]
        )
        try:
            model.fit(x_train, y_train)
            predictions = model.predict(x_test)
            probabilities = model.predict_proba(x_test)[:, 1]
        except Exception as exc:
            return {
                "status": "training_failed",
                "message": f"模型训练失败：{type(exc).__name__}: {exc}",
                **diagnostics,
            }

        model_accuracy = float(accuracy_score(y_test, predictions))
        baseline_accuracy = diagnostics["baseline_accuracy"]
        accuracy_diff = (
            model_accuracy - baseline_accuracy
            if baseline_accuracy is not None
            else None
        )
        tn, fp, fn, tp = confusion_matrix(y_test, predictions, labels=[0, 1]).ravel()

        roc_auc = None
        if y_test.nunique() > 1:
            roc_auc = float(roc_auc_score(y_test, probabilities))

        classifier = model.named_steps["classifier"]
        coefficient_pairs = sorted(
            zip(feature_columns, classifier.coef_[0]),
            key=lambda item: abs(float(item[1])),
            reverse=True,
        )[:20]
        coefficients = {feature: float(value) for feature, value in coefficient_pairs}
        intercept = float(classifier.intercept_[0])

        model_path = Path("data/models/ml_baseline_hit_5d.joblib")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        save_format = "joblib"
        try:
            import joblib

            joblib.dump(model, model_path)
        except ImportError:
            save_format = "pickle"
            with model_path.open("wb") as file:
                pickle.dump(model, file)

        return {
            "status": "trained",
            "target": target,
            **diagnostics,
            "accuracy": model_accuracy,
            "precision": float(precision_score(y_test, predictions, zero_division=0)),
            "recall": float(recall_score(y_test, predictions, zero_division=0)),
            "roc_auc": roc_auc,
            "prediction_distribution_test": self._distribution(pd.Series(predictions)),
            "confusion_matrix": {
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            },
            "predicted_positive_rate": float(pd.Series(predictions).mean()),
            "coefficients": coefficients,
            "intercept": intercept,
            "model_vs_baseline": {
                "accuracy_diff": accuracy_diff,
                "outperformed": accuracy_diff is not None and accuracy_diff > 0,
            },
            "model_path": str(model_path),
            "save_format": save_format,
        }

    @staticmethod
    def _binary_target(series: pd.Series) -> pd.Series:
        mapping = {
            "true": 1.0, "1": 1.0, "1.0": 1.0, "yes": 1.0,
            "false": 0.0, "0": 0.0, "0.0": 0.0, "no": 0.0,
        }
        return series.astype("string").str.strip().str.lower().map(mapping).astype("float64")

    @staticmethod
    def _distribution(series: pd.Series) -> dict[str, int]:
        numeric = pd.to_numeric(series, errors="coerce")
        return {
            "0": int((numeric == 0).sum()),
            "1": int((numeric == 1).sum()),
        }

    @staticmethod
    def _mean_or_none(series: pd.Series) -> float | None:
        return None if series.empty else float(series.mean())

    @staticmethod
    def _majority_accuracy(series: pd.Series) -> float | None:
        if series.empty:
            return None
        counts = series.value_counts()
        return float(counts.max() / len(series))
