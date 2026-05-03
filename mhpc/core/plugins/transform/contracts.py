"""Slot-root contracts for transform plugins."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np
import torch
from mhpc.core.plugins.locality_context_contract import LocalityContext
from mhpc.core.train_update_context_contract import TrainUpdateContext


@runtime_checkable
class TransformBindContextLike(Protocol):
    """Protocol-only bind context surface consumed by transform plugins."""

    training_contract: str
    seed: int


@dataclass(frozen=True)
class TransformRegularizationSettings:
    """Transform-stage regularization settings payload."""

    enabled: bool
    method: str
    shrinkage: str | float
    eigen_floor_ratio: float
    min_jitter: float
    max_jitter: float
    jitter_multiplier: float


@dataclass(frozen=True)
class TransformTrainContext:
    """Transform-stage train context provided by host orchestration."""

    training_contract: Literal["OFFLINE", "STREAMING"]
    feature_dim: int
    regularization: TransformRegularizationSettings


@runtime_checkable
class TransformPlugin(Protocol):
    """Contract-only protocol for transform slot forward."""

    requires_fit_state: bool
    """Whether transform-stage training must fit and apply statistical state."""

    requires_locality_context: bool
    """Whether transform fit/infer requires explicit aligned locality metadata."""

    preserves_locality: bool
    """Whether transform semantics remain valid after a locality frontier begins."""

    def bind_params(
        self,
        *,
        params: Mapping[str, Any],
        bind_context: TransformBindContextLike,
    ) -> None:
        """Bind full slot parameter bundle once for this plugin instance."""
        ...

    def resolve_train_context(
        self,
        *,
        training_contract: Literal["OFFLINE", "STREAMING"],
        feature_dim: int,
    ) -> TransformTrainContext:
        """Resolve plugin-local transform train context from bound params."""
        ...

    def forward_embed_transform(
        self,
        *,
        features: torch.Tensor,
        forward_modules: torch.nn.ModuleDict,
        locality_context: LocalityContext | None = None,
    ) -> torch.Tensor:
        """Run transform forward with deterministic tensor flow."""
        ...

    def train_start(
        self,
        *,
        context: TransformTrainContext,
    ) -> None:
        """Initialize or reset transform-stage training state."""
        ...

    def train_update(
        self,
        *,
        batch: np.ndarray,
        locality_context: LocalityContext | None = None,
        update_context: TrainUpdateContext | None = None,
    ) -> None:
        """Consume one transformed feature batch for transform-state fitting."""
        ...

    def train_finalize(self) -> None:
        """Finalize transform-stage fitted state after train updates."""
        ...

    def infer_transform(
        self,
        *,
        features: np.ndarray,
        stage: str,
        batch_idx: int | None = None,
        locality_context: LocalityContext | None = None,
    ) -> np.ndarray:
        """Apply inference-time transform with frozen plugin-owned state."""
        ...

    def state_export(self) -> object | None:
        """Export plugin-owned opaque transform state."""
        ...

    def state_load(
        self,
        *,
        state: object | None,
    ) -> None:
        """Load plugin-owned opaque transform state."""
        ...

__all__ = [
    "LocalityContext",
    "TransformPlugin",
    "TransformBindContextLike",
    "TrainUpdateContext",
    "TransformRegularizationSettings",
    "TransformTrainContext",
]
