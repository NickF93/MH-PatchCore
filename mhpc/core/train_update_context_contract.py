"""Shared generic train-update context contract for trainable plugin surfaces."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainUpdateContext:
    """Generic host-owned metadata for one train-update call."""

    epoch_index: int
    epoch_count: int
    batch_index: int


__all__ = ["TrainUpdateContext"]
