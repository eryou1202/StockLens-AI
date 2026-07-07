from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.ml.ml_evaluator import MLEvaluator
from src.ml.ml_model_registry import ResearchModelRegistry
from src.ml.ml_preprocess import MLPreprocessor
from src.ml.ml_schema import MLTrainRequest


class MLTrainer:
    """Train time-split baselines and register them as research-only artifacts."""

    def __init__(self, registry: ResearchModelRegistry | None = None) -> None:
        self.registry = registry or ResearchModelRegistry()
        self.preprocessor = MLPreprocessor()
        self.evaluator = MLEvaluator()

    def train(self, request: MLTrainRequest) -> dict[str, Any]:
        dataset_path = Path(request.dataset_path)
        if not dataset_path.exists():
            return {"status": "dataset_missing", "message": f"dataset not found: {dataset_path}"}
        frame = pd.read_csv(dataset_path, encoding="utf-8-sig")
        if request.target not in frame.columns:
            return {"status": "target_missing", "message": f"target not found: {request.target}"}
        if "as_of_date" not in frame.columns:
            return {"status": "date_missing", "message": "dataset has no as_of_date column"}
        if request.train_end >= request.valid_start:
            return {"status": "invalid_split", "message": "train_end must be earlier than valid_start"}

        task = self._task_type(request.target)
        if task == "classification" and request.model_type != "logistic":
            return {"status": "model_target_mismatch", "message": "hit target requires logistic model"}
        if task == "regression" and request.model_type != "random_forest_regressor":
            return {
                "status": "model_target_mismatch",
                "message": "future_return target requires random_forest_regressor",
            }
        dates = pd.to_datetime(frame["as_of_date"], errors="coerce")
        target = pd.to_numeric(frame[request.target], errors="coerce")
        train_mask = (dates <= pd.Timestamp(request.train_end)) & target.notna()
        valid_mask = (
            (dates >= pd.Timestamp(request.valid_start))
            & (dates <= pd.Timestamp(request.valid_end))
            & target.notna()
        )
        train = frame.loc[train_mask].copy()
        valid = frame.loc[valid_mask].copy()
        if train.empty or valid.empty:
            return {
                "status": "insufficient_split",
                "message": f"time split has train={len(train)}, valid={len(valid)} usable rows",
            }

        features = self.preprocessor.select_features(train, request.target)
        if not features:
            return {"status": "no_features", "message": "no safe numeric feature columns found"}
        x_train = self.preprocessor.numeric_frame(train, features)
        x_valid = self.preprocessor.numeric_frame(valid, features)
        y_train = pd.to_numeric(train[request.target], errors="coerce")
        y_valid = pd.to_numeric(valid[request.target], errors="coerce")
        if task == "classification":
            y_train = y_train.astype(int)
            y_valid = y_valid.astype(int)
            if y_train.nunique() < 2:
                return {
                    "status": "insufficient_train_classes",
                    "message": "training window contains only one target class",
                }

        pipeline = self.preprocessor.build_pipeline(request.model_type)
        try:
            pipeline.fit(x_train, y_train)
            predictions = pipeline.predict(x_valid)
            if task == "classification":
                probabilities = pipeline.predict_proba(x_valid)[:, 1]
                metrics = self.evaluator.evaluate_classification(
                    valid, request.target, probabilities, predictions
                )
            else:
                metrics = self.evaluator.evaluate_regression(valid, request.target, predictions)
        except Exception as exc:
            return {"status": "training_failed", "message": f"{type(exc).__name__}: {exc}"}

        model_id = self.registry.new_model_id()
        model_path = self.registry.model_path(model_id)
        created_at = datetime.now().isoformat(timespec="seconds")
        package = {
            "pipeline": pipeline,
            "features": features,
            "target": request.target,
            "model_type": request.model_type,
            "status": "research",
            "created_at": created_at,
            "metrics": metrics,
        }
        joblib.dump(package, model_path)
        horizon = self._horizon(request.target)
        train_dates = pd.to_datetime(train["as_of_date"], errors="coerce")
        record = {
            "model_id": model_id,
            "model_name": request.model_name,
            "model_type": request.model_type,
            "target": request.target,
            "target_horizon": horizon,
            "dataset_path": str(dataset_path),
            "train_start": train_dates.min().date().isoformat(),
            "train_end": request.train_end.date().isoformat(),
            "valid_start": request.valid_start.date().isoformat(),
            "valid_end": request.valid_end.date().isoformat(),
            "features_json": self.registry.dumps(features),
            "metrics_json": self.registry.dumps(metrics),
            "model_path": str(model_path),
            "status": "research",
            "created_at": created_at,
            "notes": request.notes,
        }
        self.registry.register(record)
        return {
            "status": "trained",
            "model_id": model_id,
            "model_path": str(model_path),
            "registry_path": str(self.registry.database_path),
            "train_samples": int(len(train)),
            "valid_samples": int(len(valid)),
            "feature_count": len(features),
            "features": features,
            "metrics": metrics,
            "research_only": True,
        }

    @staticmethod
    def _task_type(target: str) -> str:
        if target.startswith("hit_"):
            return "classification"
        if target.startswith("future_return_"):
            return "regression"
        raise ValueError("target must start with hit_ or future_return_")

    @staticmethod
    def _horizon(target: str) -> int | None:
        match = re.search(r"_(\d+)d$", target)
        return int(match.group(1)) if match else None
