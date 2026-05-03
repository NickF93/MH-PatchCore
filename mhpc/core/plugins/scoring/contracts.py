"""Scoring slot contract boundary."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np
import torch
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
class ScoringBindContextLike(Protocol):
    """Protocol-only bind context surface consumed by scoring plugins."""

    training_contract: str
    seed: int


@runtime_checkable
class PatchMakerLike(Protocol):
    """Protocol-only patch-maker surface consumed by scoring plugins."""

    def unpatch_scores(
        self,
        x: torch.Tensor,
        batchsize: int,
    ) -> torch.Tensor:
        """Reshape flat patch scores back to batch-major grids."""
        ...

    def score(
        self,
        x: torch.Tensor | np.ndarray,
    ) -> torch.Tensor | np.ndarray:
        """Reduce per-patch scores to one score per image."""
        ...


@dataclass(frozen=True)
class ScoringRuntimeControls:
    """Scoring controls resolved from plugin-bound slot params."""

    patch_scoring_mode: str
    paper_reweight_num_nn: int
    pni_prototype_source: str
    pni_train_view_policy: str
    pni_neighborhood_kernel_size: int
    pni_neighborhood_use_relative: bool
    pni_prior_mix_gamma: float
    pni_position_laplace_alpha: float
    pni_faithful_prior_threshold: float
    pni_faithful_distance_scale: float
    pni_assignment_chunk_size: int
    pni_prototype_chunk_size: int
    pni_topk_k: int = 5
    pni_topk_temperature: float = 1.0
    pni_topp_p: float = 0.90
    pni_topp_max_k: int = 32


@runtime_checkable
class ScoringSegmentor(Protocol):
    """Contract-only protocol for patch-score -> segmentation conversion."""

    def convert_to_segmentation(
        self,
        patch_scores: np.ndarray | torch.Tensor,
    ) -> list[np.ndarray]:
        """Convert patch-score grids to resized/smoothed segmentation maps."""
        ...


@runtime_checkable
class ScoringNNMethod(Protocol):
    """Protocol for nearest-neighbor backend used by NN-based scoring logic."""

    def run(
        self,
        n_nearest_neighbours: int,
        query_features: np.ndarray,
        index_features: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Execute nearest-neighbor query and return distances/indices."""
        ...


@runtime_checkable
class ScoringAnomalyScorer(Protocol):
    """Protocol for anomaly scorer state consumed by scoring plugins."""

    @property
    def detection_features(self) -> MemoryBankPayload | None:
        """Return the currently loaded/fitted detection-feature payload."""
        ...

    @property
    def nn_method(self) -> ScoringNNMethod | None:
        """Return the optional NN backend for NN-based scoring paths."""
        ...


@runtime_checkable
class ScoringPlugin(Protocol):
    """Contract-only protocol for predict-time scoring slot."""

    supports_train: bool
    supports_inference: bool
    requires_locality_context: bool
    preserves_locality: bool
    requires_patch_scoring_state: bool
    """Whether fit-time auxiliary patch-scoring state must be materialized."""

    def bind_params(
        self,
        *,
        params: Mapping[str, Any],
        bind_context: ScoringBindContextLike,
    ) -> None:
        """Bind full slot parameter bundle once for this plugin instance."""
        ...

    def resolve_scoring_controls(self) -> ScoringRuntimeControls:
        """Resolve plugin-local scoring controls from bound params."""
        ...

    def create_segmentor(
        self,
        *,
        device: torch.device | str,
        target_size: int | tuple[int, int],
    ) -> ScoringSegmentor:
        """Create plugin-owned runtime segmentor for inference masks."""
        ...

    def aux_state_fit_start(
        self,
        *,
        memory_bank: MemoryBankPayload,
    ) -> object | None:
        """Create plugin-owned fit-time aux-state runtime from bank payloads."""
        ...

    def aux_state_fit_update(
        self,
        *,
        fit_state: object | None,
        features: np.ndarray,
        batch_size: int,
        patch_shape: tuple[int, int],
        locality_context: LocalityContext | None = None,
    ) -> object | None:
        """Consume one fit-time update and return updated plugin-owned aux state."""
        ...

    def aux_state_fit_finalize(
        self,
        *,
        fit_state: object | None,
    ) -> object | None:
        """Finalize plugin-owned fit-time aux-state runtime to checkpointable state."""
        ...

    def aux_state_validate_loaded(
        self,
        *,
        state: object | None,
    ) -> None:
        """Validate loaded checkpoint aux-state payload for active scoring plugin."""
        ...

    def score(
        self,
        *,
        features: np.ndarray,
        patch_scores: np.ndarray,
        query_distances: np.ndarray,
        query_nns: np.ndarray,
        distance_query: DistanceQueryPayload | None = None,
        patch_shape: tuple[int, int],
        batchsize: int,
        patch_maker: PatchMakerLike,
        anomaly_scorer: ScoringAnomalyScorer,
        patch_scoring_mode: str,
        patch_scoring_state: object | None,
        paper_reweight_num_nn: int,
        locality_context: LocalityContext | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return image scores and effective patch scores with deterministic semantics."""
        ...


__all__ = [
    "DistanceQueryPayload",
    "MemoryBankPayload",
    "PatchMakerLike",
    "ScoringAnomalyScorer",
    "ScoringBindContextLike",
    "ScoringNNMethod",
    "LocalityContext",
    "ScoringPlugin",
    "ScoringRuntimeControls",
    "ScoringSegmentor",
    "StructuredGlobalDensityBank",
    "StructuredGlobalDensityQueryResult",
    "StructuredGlobalNNBank",
    "StructuredLocalMemoryBank",
    "StructuredLocalPositionBank",
    "StructuredLocalPositionQueryResult",
    "StructuredLocalQueryResult",
]
