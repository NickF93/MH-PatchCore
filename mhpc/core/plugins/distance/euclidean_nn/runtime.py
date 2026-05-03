"""Plugin-local NN runtime for ``euclidean_nn`` distance plugin."""

from __future__ import annotations

import os
import pickle  # nosec B403
from typing import Any, Protocol

import faiss  # type: ignore[import-untyped]
import numpy as np

from ..contracts import MemoryBankPayload, StructuredGlobalNNBank


def _normalize_global_features(features: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(features, dtype=np.float32))


def _normalize_global_memory_bank(feature: MemoryBankPayload) -> np.ndarray:
    if not isinstance(feature, np.ndarray):
        raise TypeError(
            "NearestNeighbourScorer supports only global ndarray detection features."
        )
    return _normalize_global_features(feature)


def _normalize_structured_global_nn_bank(
    feature: StructuredGlobalNNBank,
) -> StructuredGlobalNNBank:
    normalized_features = _normalize_global_features(feature.features)
    if feature.self_distances is None:
        normalized_self_distances = None
    else:
        normalized_self_distances = np.ascontiguousarray(
            np.asarray(feature.self_distances, dtype=np.float64)
        )
        if normalized_self_distances.ndim != 1:
            raise ValueError(
                "NearestNeighbourScorer requires 1D self_distances for "
                "StructuredGlobalNNBank detection features."
            )
        if normalized_self_distances.shape[0] != normalized_features.shape[0]:
            raise ValueError(
                "NearestNeighbourScorer requires self_distances to align with "
                "StructuredGlobalNNBank features rows."
            )
    return StructuredGlobalNNBank(
        features=normalized_features,
        self_distances=normalized_self_distances,
    )


def _normalize_detection_payload(
    feature: MemoryBankPayload,
) -> tuple[MemoryBankPayload, np.ndarray]:
    if isinstance(feature, StructuredGlobalNNBank):
        normalized_bank = _normalize_structured_global_nn_bank(feature)
        return normalized_bank, normalized_bank.features
    normalized_array = _normalize_global_memory_bank(feature)
    return normalized_array, normalized_array


class _DistanceNNBackend(Protocol):
    def run(
        self,
        n_nearest_neighbours: int,
        query_features: np.ndarray,
        index_features: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Execute nearest-neighbour query and return distances/indices."""
        ...

    def fit(self, features: np.ndarray) -> None:
        """Fit one global nearest-neighbour search index."""
        ...

    def save(self, filename: str) -> None:
        """Persist the nearest-neighbour backend to disk."""
        ...

    def load(self, filename: str) -> None:
        """Load the nearest-neighbour backend from disk."""
        ...

    def reset_index(self) -> None:
        """Reset in-memory backend state."""
        ...


class FaissNN:
    def __init__(self, on_gpu: bool = False, num_workers: int = 4) -> None:
        """FAISS nearest-neighbour search runtime."""
        faiss.omp_set_num_threads(num_workers)
        self.on_gpu = on_gpu
        self.search_index: Any | None = None

    def _gpu_cloner_options(self) -> Any:
        return faiss.GpuClonerOptions()

    def _index_to_gpu(self, index: Any) -> Any:
        if self.on_gpu:
            return faiss.index_cpu_to_gpu(
                faiss.StandardGpuResources(), 0, index, self._gpu_cloner_options()
            )
        return index

    def _index_to_cpu(self, index: Any) -> Any:
        if self.on_gpu:
            return faiss.index_gpu_to_cpu(index)
        return index

    def _create_index(self, dimension: int) -> Any:
        if self.on_gpu:
            return faiss.GpuIndexFlatL2(
                faiss.StandardGpuResources(), dimension, faiss.GpuIndexFlatConfig()
            )
        return faiss.IndexFlatL2(dimension)

    def fit(self, features: np.ndarray) -> None:
        """Add features to the FAISS search index."""
        if self.search_index:
            self.reset_index()
        self.search_index = self._create_index(features.shape[-1])
        self._train(self.search_index, features)
        if self.search_index is None:
            raise RuntimeError("FAISS search index was not initialized before add().")
        self.search_index.add(features)

    def _train(self, _index: Any, _features: np.ndarray) -> None:
        return

    def run(
        self,
        n_nearest_neighbours: int,
        query_features: np.ndarray,
        index_features: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return distances and nearest-neighbour indices."""
        if index_features is None:
            if self.search_index is None:
                raise RuntimeError("FAISS search index is not initialized.")
            return self.search_index.search(query_features, n_nearest_neighbours)

        search_index = self._create_index(index_features.shape[-1])
        self._train(search_index, index_features)
        search_index.add(index_features)
        return search_index.search(query_features, n_nearest_neighbours)

    def save(self, filename: str) -> None:
        faiss.write_index(self._index_to_cpu(self.search_index), filename)

    def load(self, filename: str) -> None:
        self.search_index = self._index_to_gpu(faiss.read_index(filename))

    def reset_index(self) -> None:
        if self.search_index:
            self.search_index.reset()
            self.search_index = None


class _BaseMerger:
    @staticmethod
    def _reduce(features: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def merge(self, features: list[np.ndarray]) -> np.ndarray:
        reduced_features = [self._reduce(feature) for feature in features]
        return np.concatenate(reduced_features, axis=1)


class AverageMerger(_BaseMerger):
    @staticmethod
    def _reduce(features: np.ndarray) -> np.ndarray:
        return features.reshape([features.shape[0], features.shape[1], -1]).mean(axis=-1)


class ConcatMerger(_BaseMerger):
    @staticmethod
    def _reduce(features: np.ndarray) -> np.ndarray:
        return features.reshape(len(features), -1)


class NearestNeighbourScorer:
    def __init__(self, n_nearest_neighbours: int, nn_method: Any | None = None) -> None:
        """Nearest-neighbour anomaly scorer."""
        self.feature_merger = ConcatMerger()
        self.n_nearest_neighbours = n_nearest_neighbours
        if nn_method is None:
            self.nn_method: _DistanceNNBackend = FaissNN(False, 4)
        else:
            self.nn_method = nn_method
        self.detection_features: MemoryBankPayload | None = None
        self.imagelevel_nn = lambda query: self.nn_method.run(n_nearest_neighbours, query)
        self.pixelwise_nn = lambda query, index: self.nn_method.run(1, query, index)

    def fit(self, detection_features: list[MemoryBankPayload]) -> None:
        if not detection_features:
            raise ValueError("NearestNeighbourScorer.fit() requires detection features.")
        normalized_payloads = [
            _normalize_detection_payload(feature) for feature in detection_features
        ]
        structured_payloads = [
            payload
            for payload, _ in normalized_payloads
            if isinstance(payload, StructuredGlobalNNBank)
        ]
        if structured_payloads:
            if len(normalized_payloads) != 1:
                raise ValueError(
                    "NearestNeighbourScorer.fit() requires exactly one "
                    "StructuredGlobalNNBank payload when using structured "
                    "global NN detection features."
                )
            structured_detection_payload = structured_payloads[0]
            detection_payload: MemoryBankPayload = structured_detection_payload
            merged_detection_features = structured_detection_payload.features
        else:
            global_detection_features = [
                feature_array for _, feature_array in normalized_payloads
            ]
            merged_detection_features = self.feature_merger.merge(global_detection_features)
            detection_payload = merged_detection_features
        self.detection_features = detection_payload
        self.nn_method.fit(merged_detection_features)

    def predict(
        self,
        query_features: list[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        query_feature_array = _normalize_global_features(
            self.feature_merger.merge(query_features)
        )
        query_distances, query_nns = self.imagelevel_nn(query_feature_array)
        anomaly_scores = np.mean(query_distances, axis=-1)
        return anomaly_scores, query_distances, query_nns

    @staticmethod
    def _detection_file(folder: str, prepend: str = "") -> str:
        return os.path.join(folder, prepend + "nnscorer_features.pkl")

    @staticmethod
    def _index_file(folder: str, prepend: str = "") -> str:
        return os.path.join(folder, prepend + "nnscorer_search_index.faiss")

    @staticmethod
    def _save(filename: str, features: Any) -> None:
        if features is None:
            return
        with open(filename, "wb") as save_file:
            pickle.dump(features, save_file, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def _load(filename: str) -> Any:
        with open(filename, "rb") as load_file:
            # Local checkpoint files are trusted in this repository workflow.
            return pickle.load(load_file)  # nosec B301

    def save(
        self,
        save_folder: str,
        save_features_separately: bool = False,
        prepend: str = "",
    ) -> None:
        self.nn_method.save(self._index_file(save_folder, prepend))
        if save_features_separately:
            self._save(
                self._detection_file(save_folder, prepend),
                self.detection_features,
            )

    def save_and_reset(self, save_folder: str) -> None:
        self.save(save_folder)
        self.nn_method.reset_index()

    def load(self, load_folder: str, prepend: str = "") -> None:
        self.nn_method.load(self._index_file(load_folder, prepend))
        if os.path.exists(self._detection_file(load_folder, prepend)):
            self.detection_features = self._load(
                self._detection_file(load_folder, prepend)
            )


__all__ = [
    "faiss",
    "FaissNN",
    "_BaseMerger",
    "AverageMerger",
    "ConcatMerger",
    "NearestNeighbourScorer",
]
