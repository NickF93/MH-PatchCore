"""Generic stage lifecycle contracts for stage-agnostic orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class StageLifecycleSelection:
    """Host-owned lifecycle metadata resolved by orchestration."""

    stage_name: str
    role: str
    trainable: bool
    fit_epochs: int = 1


@runtime_checkable
class StageLifecycle(Protocol):
    """Generic lifecycle contract used by train/inference orchestrators."""

    def train_start(self) -> None:
        """Initialize or reset stage-owned train-time state."""
        ...

    def train_update(self, *, batch: object) -> None:
        """Consume one train batch/update payload."""
        ...

    def train_finalize(self) -> None:
        """Finalize stage-owned train-time state after updates."""
        ...

    def infer(self, *, batch: object) -> object:
        """Run inference-time stage transformation/query on one payload."""
        ...

    def state_export(self) -> object | None:
        """Export stage-owned opaque state payload."""
        ...

    def state_load(self, *, state: object | None) -> None:
        """Load stage-owned opaque state payload."""
        ...


def validate_stage_lifecycle_object(
    lifecycle: object,
    *,
    stage_name: str,
) -> None:
    """Fail fast when an object does not satisfy the StageLifecycle contract."""

    if not isinstance(lifecycle, StageLifecycle):
        raise TypeError(
            "Lifecycle object does not satisfy StageLifecycle contract: "
            f"stage='{stage_name}' type='{type(lifecycle).__name__}'"
        )

