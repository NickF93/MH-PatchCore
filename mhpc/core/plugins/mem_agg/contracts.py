"""Slot-root contracts for memory-aggregation plugins."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np
import torch
from mhpc.core.plugins.locality_context_contract import LocalityContext
from mhpc.core.train_update_context_contract import TrainUpdateContext
from mhpc.core.plugins.locality_state_contract import (
    MemoryBankPayload,
    StructuredLocalMemoryBank,
    StructuredLocalPositionBank,
)


@runtime_checkable
class MemAggBindContextLike(Protocol):
    """Protocol-only bind context surface consumed by mem_agg plugins."""

    training_contract: str
    seed: int


@dataclass(frozen=True)
class MemAggRuntimeContext:
    """Generic host-owned runtime facts for mem_agg runtime creation."""

    training_contract: str
    device: torch.device
    feature_count: int | None = None


@dataclass(frozen=True)
class AggregationRuntimeMetadata:
    """Generic host-readable metadata exported by mem_agg runtime state."""

    reference_limit: int | None = None
    enforce_reference_limit: bool = False


@runtime_checkable
class AggregationRuntimeState(Protocol):
    """Protocol for mutable aggregation runtime state."""

    def update(
        self,
        features: np.ndarray,
        locality_context: LocalityContext | None = None,
        update_context: TrainUpdateContext | None = None,
    ) -> None:
        """Consume one batch of transformed embedding vectors."""
        ...

    def get_centroids(self) -> MemoryBankPayload:
        """Finalize and return a global or structured local memory-bank payload."""
        ...

    def export_state(self) -> dict[str, object]:
        """Export strategy-specific state payload."""
        ...

    def runtime_metadata(self) -> AggregationRuntimeMetadata:
        """Expose generic host-readable runtime metadata."""
        ...


@runtime_checkable
class MemoryAggregationPlugin(Protocol):
    """Contract-only protocol for fit-time memory aggregation wiring."""

    requires_locality_context: bool
    preserves_locality: bool

    def bind_params(
        self,
        *,
        params: Mapping[str, Any],
        bind_context: MemAggBindContextLike,
    ) -> None:
        """Bind full slot parameter bundle once for this plugin instance."""
        ...

    def create_runtime_state(
        self,
        *,
        runtime_context: MemAggRuntimeContext,
    ) -> AggregationRuntimeState:
        """Build a fresh aggregation runtime state for one fit run."""
        ...

__all__ = [
    "AggregationRuntimeMetadata",
    "AggregationRuntimeState",
    "LocalityContext",
    "MemAggRuntimeContext",
    "MemAggBindContextLike",
    "MemoryBankPayload",
    "TrainUpdateContext",
    "MemoryAggregationPlugin",
    "StructuredLocalMemoryBank",
    "StructuredLocalPositionBank",
]
