"""Distance slot contract boundary."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

import numpy as np
from mhpc.core.plugins.locality_context_contract import LocalityContext
from mhpc.core.plugins.locality_state_contract import (
    DistanceQueryPayload,
    MemoryBankPayload,
    StructuredGlobalDensityBank,
    StructuredGlobalDensityQueryResult,
    StructuredGlobalNNBank,
    StructuredLocalMemoryBank,
    StructuredLocalPositionBank,
    StructuredLocalPositionQueryResult,
    StructuredLocalQueryResult,
)


@runtime_checkable
class DistanceBindContextLike(Protocol):
    """Protocol-only bind context surface consumed by distance plugins."""

    training_contract: str
    seed: int


@runtime_checkable
class DistanceNNMethod(Protocol):
    """Protocol for NN backend used by NN-based distance/scoring logic."""

    def run(
        self,
        n_nearest_neighbours: int,
        query_features: np.ndarray,
        index_features: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Execute nearest-neighbour query and return distances/indices."""
        ...


@runtime_checkable
class DistanceAnomalyScorer(Protocol):
    """Protocol for scorer runtime required across core fit/infer paths."""

    @property
    def detection_features(self) -> MemoryBankPayload | None:
        """Return the currently loaded/fitted detection-feature payload."""
        ...

    @property
    def nn_method(self) -> DistanceNNMethod | None:
        """Return the optional NN backend for NN-based distance families."""
        ...

    def predict(
        self,
        query_features: list[np.ndarray],
    ) -> DistanceQueryPayload:
        """Return global or structured local nearest-neighbour outputs."""
        ...

    def fit(self, detection_features: list[MemoryBankPayload]) -> None:
        """Fit runtime search index for one detection feature payload."""
        ...

    def save(
        self,
        save_folder: str,
        save_features_separately: bool = False,
        prepend: str = "",
    ) -> None:
        """Persist runtime index/features to checkpoint files."""
        ...

    def load(self, load_folder: str, prepend: str = "") -> None:
        """Load runtime index/features from checkpoint files."""
        ...


@runtime_checkable
class DistancePlugin(Protocol):
    """Contract-only protocol for anomaly scorer query path."""

    supports_train: bool
    supports_inference: bool
    requires_locality_context: bool
    preserves_locality: bool

    def bind_params(
        self,
        *,
        params: Mapping[str, Any],
        bind_context: DistanceBindContextLike,
    ) -> None:
        """Bind full slot parameter bundle once for this plugin instance."""
        ...

    def resolve_num_neighbors(self) -> int:
        """Resolve plugin-local nearest-neighbour `k` from bound params."""
        ...

    def create_anomaly_scorer(
        self,
        *,
        n_nearest_neighbours: int,
    ) -> DistanceAnomalyScorer:
        """Create plugin-owned anomaly scorer runtime for this slot."""
        ...

    def query(
        self,
        *,
        anomaly_scorer: DistanceAnomalyScorer,
        features: np.ndarray,
        locality_context: LocalityContext | None = None,
    ) -> DistanceQueryPayload:
        """Query scorer with global or structured local distance outputs."""
        ...

__all__ = [
    "DistanceAnomalyScorer",
    "DistanceBindContextLike",
    "DistanceNNMethod",
    "DistanceQueryPayload",
    "DistancePlugin",
    "LocalityContext",
    "MemoryBankPayload",
    "StructuredGlobalDensityBank",
    "StructuredGlobalDensityQueryResult",
    "StructuredGlobalNNBank",
    "StructuredLocalMemoryBank",
    "StructuredLocalPositionBank",
    "StructuredLocalPositionQueryResult",
    "StructuredLocalQueryResult",
]
