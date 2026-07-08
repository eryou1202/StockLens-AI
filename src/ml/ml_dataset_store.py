from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.ml.ml_schema import CONTEXT_FEATURE_COLUMNS, MLResearchSample
from src.ml.ml_relative_labels import RelativeTargetLabelBuilder


class MLDatasetStore:
    def save(self, samples: list[MLResearchSample], output_path: str) -> pd.DataFrame:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        columns = list(MLResearchSample.model_fields)
        columns.remove("metadata")
        include_context = any(
            bool(sample.metadata.get("include_context_features")) for sample in samples
        )
        if not include_context:
            columns = [column for column in columns if column not in CONTEXT_FEATURE_COLUMNS]
        frame = pd.DataFrame(
            [sample.model_dump(mode="json", exclude={"metadata"}) for sample in samples],
            columns=columns,
        )
        frame = RelativeTargetLabelBuilder.apply(frame)
        frame.to_csv(path, index=False, encoding="utf-8-sig")
        return frame

    @staticmethod
    def load(path: str) -> pd.DataFrame:
        return pd.read_csv(Path(path), encoding="utf-8-sig")
