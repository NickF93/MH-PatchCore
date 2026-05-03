"""Host-owned structured locality-preserving downstream state contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np

LocalPosition: TypeAlias = tuple[int, int]
GlobalDistanceQueryTriplet: TypeAlias = tuple[np.ndarray, np.ndarray, np.ndarray]


@dataclass(frozen=True)
class StructuredGlobalDensityBank:
    """Structured global density-bank payload for component-based models."""

    model_family: str
    component_weights: np.ndarray
    component_means: np.ndarray
    component_variances: np.ndarray
    component_effective_counts: np.ndarray
    feature_dim: int
    covariance_type: str
    regularization: float
    seen_samples: int
    update_count: int
    is_initialized: bool


@dataclass(frozen=True)
class StructuredGlobalNNBank:
    """Structured global NN-bank payload with optional self-distance metadata."""

    features: np.ndarray
    self_distances: np.ndarray | None = None


@dataclass(frozen=True)
class StructuredLocalPositionBank:
    """Per-position bank payload owned by a locality-preserving plugin chain."""

    position: LocalPosition
    features: np.ndarray


@dataclass(frozen=True)
class StructuredLocalMemoryBank:
    """Structured memory-bank payload keyed by aligned patch position."""

    patch_shape: tuple[int, int]
    position_banks: tuple[StructuredLocalPositionBank, ...]
    flatten_order: Literal["image_major_row_major"] = "image_major_row_major"


@dataclass(frozen=True)
class StructuredLocalPositionQueryResult:
    """Per-position nearest-neighbour outputs with local bank indexing."""

    position: LocalPosition
    patch_scores: np.ndarray
    query_distances: np.ndarray
    query_nns: np.ndarray


@dataclass(frozen=True)
class StructuredLocalQueryResult:
    """Structured query payload preserving same-position-only comparisons."""

    patch_shape: tuple[int, int]
    position_results: tuple[StructuredLocalPositionQueryResult, ...]
    flatten_order: Literal["image_major_row_major"] = "image_major_row_major"


@dataclass(frozen=True)
class StructuredGlobalDensityQueryResult:
    """Structured global query payload for density-model downstream stages."""

    patch_scores: np.ndarray
    component_ids: np.ndarray
    component_log_probs: np.ndarray
    component_posteriors: np.ndarray


MemoryBankPayload: TypeAlias = (
    np.ndarray
    | StructuredGlobalNNBank
    | StructuredGlobalDensityBank
    | StructuredLocalMemoryBank
)
DistanceQueryPayload: TypeAlias = (
    GlobalDistanceQueryTriplet
    | StructuredGlobalDensityQueryResult
    | StructuredLocalQueryResult
)


__all__ = [
    "DistanceQueryPayload",
    "GlobalDistanceQueryTriplet",
    "LocalPosition",
    "MemoryBankPayload",
    "StructuredGlobalDensityBank",
    "StructuredGlobalNNBank",
    "StructuredGlobalDensityQueryResult",
    "StructuredLocalMemoryBank",
    "StructuredLocalPositionBank",
    "StructuredLocalPositionQueryResult",
    "StructuredLocalQueryResult",
]
