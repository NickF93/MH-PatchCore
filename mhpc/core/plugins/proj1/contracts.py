"""Slot-local contracts for projector-1 plugins."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np
import torch
from mhpc.core.plugins.locality_context_contract import LocalityContext
from mhpc.core.train_update_context_contract import TrainUpdateContext


@runtime_checkable
class Projector1BindContextLike(Protocol):
    """Protocol-only bind context surface consumed by proj1 plugins."""

    training_contract: str
    seed: int


@dataclass(frozen=True)
class Projector1TrainContext:
    """Projector-1 train context provided by host orchestration."""

    training_contract: Literal["OFFLINE", "STREAMING"]
    feature_dim: int
    device: torch.device


@runtime_checkable
class Projector1Plugin(Protocol):
    """Contract-only protocol for STEP_5 projector-1 forward behavior."""

    requires_fit_state: bool
    """Whether proj1 requires fitted state before inference-time application."""

    requires_locality_context: bool
    preserves_locality: bool

    def bind_params(
        self,
        *,
        params: Mapping[str, Any],
        bind_context: Projector1BindContextLike,
    ) -> None:
        """Bind full slot parameter bundle once for this plugin instance."""
        ...

    def forward_embed_projector1(
        self,
        *,
        features: torch.Tensor,
        forward_modules: torch.nn.ModuleDict,
        locality_context: LocalityContext | None = None,
    ) -> torch.Tensor:
        """Run projector-1 forward with deterministic tensor flow."""
        ...

    def resolve_train_context(
        self,
        *,
        training_contract: Literal["OFFLINE", "STREAMING"],
        feature_dim: int,
        device: torch.device,
    ) -> Projector1TrainContext:
        """Resolve plugin-local proj1 train context from bound params."""
        ...

    def train_start(
        self,
        *,
        context: Projector1TrainContext,
    ) -> None:
        """Initialize or reset proj1 training state."""
        ...

    def train_update(
        self,
        *,
        batch: np.ndarray,
        locality_context: LocalityContext | None = None,
        update_context: TrainUpdateContext | None = None,
    ) -> None:
        """Consume one proj1 input batch for projector-state fitting."""
        ...

    def train_finalize(self) -> None:
        """Finalize proj1 fitted state after train updates."""
        ...

    def infer_projector1(
        self,
        *,
        features: np.ndarray,
        stage: str,
        batch_idx: int | None = None,
        locality_context: LocalityContext | None = None,
    ) -> np.ndarray:
        """Apply inference-time proj1 with frozen plugin-owned state."""
        ...

    def state_export(self) -> object | None:
        """Export plugin-owned opaque proj1 state."""
        ...

    def state_load(
        self,
        *,
        state: object | None,
    ) -> None:
        """Load plugin-owned opaque proj1 state."""
        ...

__all__ = [
    "Projector1BindContextLike",
    "Projector1Plugin",
    "Projector1TrainContext",
    "TrainUpdateContext",
]
