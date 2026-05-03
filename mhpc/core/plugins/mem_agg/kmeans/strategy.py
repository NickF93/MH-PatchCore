"""KMeans memory aggregation strategies."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans  # type: ignore[import-untyped]

from ..contracts import AggregationRuntimeMetadata


class BatchKMeansAggregationStrategy:
    """Batch KMeans aggregation for vanilla mode."""

    def __init__(self, n_clusters: int, *, random_state: int = 0) -> None:
        self._n_clusters = int(n_clusters)
        self._random_state = int(random_state)
        self._batches: list[np.ndarray] = []
        self._kmeans: KMeans | None = None
        self._runtime_metadata = AggregationRuntimeMetadata(
            reference_limit=self._n_clusters,
            enforce_reference_limit=False,
        )

    @property
    def name(self) -> str:
        return "KMEANS_BATCH"

    @property
    def is_streaming(self) -> bool:
        return False

    @property
    def supports_multi_pass(self) -> bool:
        return False

    def update(
        self,
        batch: np.ndarray,
        locality_context: object | None = None,
        update_context: object | None = None,
    ) -> None:
        _ = update_context
        del locality_context
        self._batches.append(np.asarray(batch))

    def get_centroids(self) -> np.ndarray:
        if not self._batches:
            raise RuntimeError("BatchKMeansAggregationStrategy received no batches.")
        features = np.concatenate(self._batches, axis=0)
        kmeans_model = KMeans(n_clusters=self._n_clusters, n_init="auto", random_state=self._random_state)
        kmeans_model.fit(features)
        self._kmeans = kmeans_model
        return np.asarray(kmeans_model.cluster_centers_)

    def export_state(self) -> dict[str, Any]:
        return {"kmeans_model": self._kmeans}

    def runtime_metadata(self) -> AggregationRuntimeMetadata:
        return self._runtime_metadata


class StreamingMiniBatchKMeansAggregationStrategy:
    """Streaming MiniBatchKMeans aggregation for streaming mode."""

    def __init__(
        self,
        n_clusters: int,
        minibatch_size: int = 256,
        *,
        random_state: int = 0,
        enforce_reference_limit: bool = True,
    ) -> None:
        self._n_clusters = int(n_clusters)
        self._minibatch_size = int(minibatch_size)
        self._random_state = int(random_state)
        self._kmeans = MiniBatchKMeans(
            n_clusters=self._n_clusters,
            batch_size=self._minibatch_size,
            n_init="auto",
            random_state=self._random_state,
        )
        self._buffer: list[np.ndarray] = []
        self._buffer_size = 0
        self._min_buffer_size = max(self._n_clusters, self._minibatch_size)
        self._seen_samples = 0
        self._runtime_metadata = AggregationRuntimeMetadata(
            reference_limit=self._n_clusters,
            enforce_reference_limit=bool(enforce_reference_limit),
        )

    @property
    def name(self) -> str:
        return "KMEANS_STREAMING"

    @property
    def is_streaming(self) -> bool:
        return True

    @property
    def supports_multi_pass(self) -> bool:
        return True

    def update(
        self,
        batch: np.ndarray,
        locality_context: object | None = None,
        update_context: object | None = None,
    ) -> None:
        _ = update_context
        del locality_context
        batch_np = np.asarray(batch)
        if batch_np.ndim != 2:
            raise ValueError(
                f"Expected 2D feature batch [N,D], got shape={batch_np.shape}"
            )
        self._buffer.append(batch_np)
        self._buffer_size += int(batch_np.shape[0])
        self._seen_samples += int(batch_np.shape[0])

        if self._buffer_size >= self._min_buffer_size:
            to_fit = np.concatenate(self._buffer, axis=0)
            self._kmeans.partial_fit(to_fit)
            self._buffer = []
            self._buffer_size = 0

    def get_centroids(self) -> np.ndarray:
        if self._seen_samples <= 0:
            raise RuntimeError(
                "StreamingMiniBatchKMeansAggregationStrategy received no batches."
            )
        if self._buffer:
            to_fit = np.concatenate(self._buffer, axis=0)
            self._kmeans.partial_fit(to_fit)
            self._buffer = []
            self._buffer_size = 0
        return np.asarray(self._kmeans.cluster_centers_)

    def export_state(self) -> dict[str, Any]:
        return {"kmeans_model": self._kmeans}

    def runtime_metadata(self) -> AggregationRuntimeMetadata:
        return self._runtime_metadata
