"""No-op reduction strategy for `feature_agg:none`."""

from __future__ import annotations

import numpy as np

from ..contracts import FeatureReductionStrategy


class NoOpFeatureReductionStrategy(FeatureReductionStrategy):
    """Identity reducer used by `feature_agg:none`."""

    @property
    def name(self) -> str:
        return "NONE"

    @property
    def requires_streaming_pass(self) -> bool:
        return False

    @property
    def supports_multi_pass(self) -> bool:
        return False

    @property
    def output_dimension(self) -> int | None:
        return None

    def update(self, batch: np.ndarray, update_context: object | None = None) -> None:
        _ = update_context
        _ = batch

    def finalize(self) -> None:
        return None

    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        return np.asarray(features)

    def transform(self, features: np.ndarray) -> np.ndarray:
        return np.asarray(features)

    def export_state(self) -> None:
        return None
