"""Streaming KCenter memory aggregation strategy."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..contracts import AggregationRuntimeMetadata


def _farthest_first_coreset(data: np.ndarray, target_size: int) -> np.ndarray:
    """Select a deterministic farthest-first coreset from a 2D feature array."""
    data_np = np.asarray(data, dtype=np.float64)
    if data_np.ndim != 2:
        raise ValueError(
            f"Expected 2D feature matrix for coreset selection, got {data_np.shape}"
        )
    if data_np.shape[0] == 0:
        raise ValueError("Cannot build a coreset from an empty feature matrix.")
    if target_size <= 0:
        raise ValueError("target_size must be > 0 for coreset selection.")

    n_samples = int(data_np.shape[0])
    k = min(int(target_size), n_samples)
    if k == n_samples:
        return data_np.copy()

    mean_vector = np.mean(data_np, axis=0, dtype=np.float64, keepdims=True)
    dist_to_mean = np.linalg.norm(data_np - mean_vector, axis=1)
    first_idx = int(np.argmax(dist_to_mean))

    selected_indices: list[int] = [first_idx]
    min_distances = np.linalg.norm(data_np - data_np[first_idx], axis=1)
    min_distances[first_idx] = 0.0

    while len(selected_indices) < k:
        next_idx = int(np.argmax(min_distances))
        selected_indices.append(next_idx)
        new_distances = np.linalg.norm(data_np - data_np[next_idx], axis=1)
        min_distances = np.minimum(min_distances, new_distances)
        min_distances[next_idx] = 0.0

    return data_np[np.asarray(selected_indices, dtype=np.int64)]


class StreamingKCenterAggregationStrategy:
    """Bounded-memory streaming k-center coverage coreset strategy."""

    def __init__(
        self,
        n_clusters: int,
        mode: str = "budgeted",
        chunk_coreset_size: int = 256,
        distance_threshold: float | None = None,
        *,
        enforce_reference_limit: bool = True,
    ) -> None:
        if n_clusters <= 0:
            raise ValueError("StreamingKCenterAggregationStrategy requires n_clusters > 0.")
        normalized_mode = mode.strip().lower()
        if normalized_mode not in {"budgeted", "merge_reduce"}:
            raise ValueError("kcenter_mode must be one of: budgeted, merge_reduce.")
        if chunk_coreset_size <= 0:
            raise ValueError("kcenter_chunk_coreset_size must be > 0.")
        if distance_threshold is not None and distance_threshold < 0.0:
            raise ValueError("kcenter_distance_threshold must be >= 0 when provided.")

        self._n_clusters = int(n_clusters)
        self._mode = normalized_mode
        self._chunk_coreset_size = int(chunk_coreset_size)
        self._distance_threshold = (
            None if distance_threshold is None else float(distance_threshold)
        )

        self._seen_samples = 0
        self._dimension: int | None = None
        self._centers: np.ndarray | None = None
        self._levels: dict[int, list[np.ndarray]] = {}
        self._redundant_index: int | None = None
        self._redundancy_threshold: float | None = None
        self._runtime_metadata = AggregationRuntimeMetadata(
            reference_limit=self._n_clusters,
            enforce_reference_limit=bool(enforce_reference_limit),
        )

    @property
    def name(self) -> str:
        return "KCENTER_STREAMING"

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
        batch_np = np.asarray(batch, dtype=np.float64)
        if batch_np.ndim != 2:
            raise ValueError(
                f"Expected 2D feature batch [N,D], got shape={batch_np.shape}"
            )
        if batch_np.shape[0] == 0:
            return
        if not np.all(np.isfinite(batch_np)):
            raise ValueError("StreamingKCenterAggregationStrategy received non-finite features.")

        if self._dimension is None:
            self._dimension = int(batch_np.shape[1])
        elif int(batch_np.shape[1]) != int(self._dimension):
            raise ValueError(
                "StreamingKCenterAggregationStrategy received inconsistent "
                f"feature dimensions: expected {self._dimension}, got {batch_np.shape[1]}."
            )

        self._seen_samples += int(batch_np.shape[0])
        if self._mode == "budgeted":
            for sample in batch_np:
                self._update_budgeted(sample)
            return

        local_size = min(
            self._n_clusters,
            self._chunk_coreset_size,
            int(batch_np.shape[0]),
        )
        local_block = _farthest_first_coreset(batch_np, target_size=local_size)
        self._push_merge_reduce_block(local_block)

    def get_centroids(self) -> np.ndarray:
        if self._seen_samples <= 0:
            raise RuntimeError("StreamingKCenterAggregationStrategy received no batches.")

        if self._mode == "budgeted":
            if self._centers is None or self._centers.shape[0] == 0:
                raise RuntimeError(
                    "StreamingKCenterAggregationStrategy produced no coverage centers."
                )
            return np.asarray(self._centers, dtype=np.float64)

        all_blocks: list[np.ndarray] = []
        for level in sorted(self._levels.keys()):
            all_blocks.extend(self._levels[level])
        if not all_blocks:
            raise RuntimeError(
                "StreamingKCenterAggregationStrategy produced no merge-reduce blocks."
            )
        merged = np.concatenate(all_blocks, axis=0)
        final_size = min(self._n_clusters, int(merged.shape[0]))
        return _farthest_first_coreset(merged, target_size=final_size)

    def export_state(self) -> dict[str, Any]:
        return {
            "kmeans_model": None,
            "stream_state": {
                "type": "kcenter",
                "mode": self._mode,
                "seen_samples": int(self._seen_samples),
                "n_clusters": int(self._n_clusters),
            },
        }

    def runtime_metadata(self) -> AggregationRuntimeMetadata:
        return self._runtime_metadata

    def _update_budgeted(self, sample: np.ndarray) -> None:
        if self._centers is None:
            self._centers = np.asarray(sample, dtype=np.float64).reshape(1, -1)
            self._recompute_redundancy()
            return

        if int(self._centers.shape[0]) < self._n_clusters:
            self._centers = np.concatenate(
                [self._centers, np.asarray(sample, dtype=np.float64).reshape(1, -1)],
                axis=0,
            )
            if int(self._centers.shape[0]) == self._n_clusters:
                self._recompute_redundancy()
            return

        distances = np.linalg.norm(self._centers - sample[None, :], axis=1)
        d_min = float(np.min(distances))

        threshold = self._distance_threshold
        if threshold is None:
            if self._redundancy_threshold is None:
                self._recompute_redundancy()
            threshold = float(self._redundancy_threshold or 0.0)
        if d_min <= threshold:
            return

        replacement_idx = (
            self._redundant_index
            if self._redundant_index is not None
            else int(np.argmin(distances))
        )
        self._centers[replacement_idx] = np.asarray(sample, dtype=np.float64)
        self._recompute_redundancy()

    def _recompute_redundancy(self) -> None:
        if self._centers is None:
            self._redundant_index = None
            self._redundancy_threshold = None
            return
        num_centers = int(self._centers.shape[0])
        if num_centers <= 1:
            self._redundant_index = 0
            self._redundancy_threshold = 0.0
            return

        distances = np.linalg.norm(
            self._centers[:, None, :] - self._centers[None, :, :],
            axis=2,
        )
        np.fill_diagonal(distances, np.inf)
        nearest_distances = np.min(distances, axis=1)
        self._redundant_index = int(np.argmin(nearest_distances))
        self._redundancy_threshold = float(nearest_distances[self._redundant_index])

    def _push_merge_reduce_block(self, block: np.ndarray) -> None:
        candidate = np.asarray(block, dtype=np.float64)
        level = 0
        while True:
            slots = self._levels.setdefault(level, [])
            slots.append(candidate)
            if len(slots) < 2:
                return

            left = slots.pop(0)
            right = slots.pop(0)
            merged = np.concatenate([left, right], axis=0)
            reduced_size = min(
                self._n_clusters,
                self._chunk_coreset_size,
                int(merged.shape[0]),
            )
            candidate = _farthest_first_coreset(merged, target_size=reduced_size)
            level += 1
