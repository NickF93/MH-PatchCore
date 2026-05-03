"""Incremental/Batch PCA strategies for feature aggregation."""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import IncrementalPCA, PCA  # type: ignore[import-untyped]

from ..contracts import FeatureReductionStrategy


class NoOpFeatureReductionStrategy(FeatureReductionStrategy):
    """Identity reducer used by `ipca` for reduction NONE."""

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


class BatchPCAReductionStrategy(FeatureReductionStrategy):
    """Batch PCA reducer for vanilla mode."""

    def __init__(self, variance_ratio: float) -> None:
        self._variance_ratio = float(variance_ratio)
        self._pca: PCA | None = None

    @property
    def name(self) -> str:
        return "PCA"

    @property
    def requires_streaming_pass(self) -> bool:
        return False

    @property
    def supports_multi_pass(self) -> bool:
        return False

    @property
    def output_dimension(self) -> int | None:
        if self._pca is None:
            return None
        return int(self._pca.n_components_)

    def update(self, batch: np.ndarray, update_context: object | None = None) -> None:
        _ = update_context
        _ = batch

    def finalize(self) -> None:
        return None

    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        pca_model = PCA(n_components=self._variance_ratio)
        transformed = pca_model.fit_transform(features)
        self._pca = pca_model
        return np.asarray(transformed)

    def transform(self, features: np.ndarray) -> np.ndarray:
        if self._pca is None:
            raise RuntimeError("BatchPCAReductionStrategy is not fitted.")
        return np.asarray(self._pca.transform(features))

    def export_state(self) -> PCA | None:
        return self._pca


class StreamingIncrementalPCAReductionStrategy(FeatureReductionStrategy):
    """Incremental PCA reducer for streaming multi-pass mode."""

    def __init__(self, variance_ratio: float) -> None:
        self._variance_ratio = float(variance_ratio)
        self._pca = IncrementalPCA(n_components=None)
        self._seen_batches = 0
        self._finalized = False
        self._output_dim: int | None = None

    @property
    def name(self) -> str:
        return "PCA"

    @property
    def requires_streaming_pass(self) -> bool:
        return True

    @property
    def supports_multi_pass(self) -> bool:
        return True

    @property
    def output_dimension(self) -> int | None:
        return self._output_dim

    def update(self, batch: np.ndarray, update_context: object | None = None) -> None:
        _ = update_context
        self._pca.partial_fit(batch)
        self._seen_batches += 1

    def finalize(self) -> None:
        if self._seen_batches <= 0:
            raise RuntimeError("StreamingIncrementalPCAReductionStrategy received no batches.")
        cumulative_variance = np.cumsum(self._pca.explained_variance_ratio_)
        n_components = int(np.argmax(cumulative_variance >= self._variance_ratio) + 1)
        self._pca.n_components = n_components
        self._pca.components_ = self._pca.components_[:n_components]
        self._pca.n_components_ = n_components
        self._output_dim = n_components
        self._finalized = True

    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        self.update(features)
        self.finalize()
        return self.transform(features)

    def transform(self, features: np.ndarray) -> np.ndarray:
        if not self._finalized:
            raise RuntimeError(
                "StreamingIncrementalPCAReductionStrategy must be finalized before transform()."
            )
        return np.asarray(self._pca.transform(features))

    def export_state(self) -> IncrementalPCA:
        return self._pca
