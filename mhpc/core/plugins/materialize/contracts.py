"""Slot-root contracts for materialize plugins."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

from mhpc.core.plugins.locality_context_contract import LocalityContext
from mhpc.core.plugins.locality_state_contract import (
    MemoryBankPayload,
    StructuredGlobalDensityBank,
    StructuredGlobalNNBank,
    StructuredLocalMemoryBank,
    StructuredLocalPositionBank,
)


@runtime_checkable
class MaterializationBindContextLike(Protocol):
    """Protocol-only bind context surface consumed by materialize plugins."""

    training_contract: str
    seed: int


@runtime_checkable
class MaterializationInputState(Protocol):
    """Neutral materialization input surface consumed by this slot."""

    def get_centroids(self) -> MemoryBankPayload:
        """Finalize and return a global or structured local memory-bank payload."""
        ...

    def export_state(self) -> dict[str, object]:
        """Export strategy-specific state payload."""
        ...


@runtime_checkable
class MaterializationPlugin(Protocol):
    """Contract-only protocol for materializing memory-bank outputs."""

    requires_locality_context: bool
    preserves_locality: bool

    def bind_params(
        self,
        *,
        params: Mapping[str, Any],
        bind_context: MaterializationBindContextLike,
    ) -> None:
        """Bind full slot parameter bundle once for this plugin instance."""
        ...

    def materialize(
        self,
        *,
        state: MaterializationInputState,
        locality_context: LocalityContext | None = None,
    ) -> tuple[MemoryBankPayload, dict[str, object]]:
        """Finalize runtime state and return materialized bank + exported state."""
        ...

__all__ = [
    "LocalityContext",
    "MaterializationBindContextLike",
    "MemoryBankPayload",
    "MaterializationInputState",
    "MaterializationPlugin",
    "StructuredGlobalDensityBank",
    "StructuredGlobalNNBank",
    "StructuredLocalMemoryBank",
    "StructuredLocalPositionBank",
]
