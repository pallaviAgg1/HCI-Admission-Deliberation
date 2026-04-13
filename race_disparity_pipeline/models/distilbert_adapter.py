from __future__ import annotations

from typing import Dict

from ..feature_extractor import FeatureSignal


class DistilBertFeatureExtractor:
    """Adapter placeholder for a future fine-tuned DistilBERT feature extractor.

    Keep this method signature stable so the API does not change when model
    inference is introduced.
    """

    def __init__(self, model_path: str | None = None) -> None:
        self.model_path = model_path

    def extract_features(self, text: str) -> Dict[str, FeatureSignal]:
        raise NotImplementedError(
            "DistilBERT adapter is not implemented yet. "
            "Use rule-based extract_features() for this phase."
        )
