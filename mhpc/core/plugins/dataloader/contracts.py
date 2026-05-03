"""Slot-local contracts for dataloader plugins."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class DataLoaderBindContextLike(Protocol):
    """Protocol-only bind context surface consumed by dataloader plugins."""

    training_contract: str
    seed: int
    repo_root: Path


@runtime_checkable
class DataLoaderPlugin(Protocol):
    """Contract-only protocol for dataset-plan and loader construction."""

    def bind_params(
        self,
        *,
        params: Mapping[str, Any],
        bind_context: DataLoaderBindContextLike,
    ) -> None:
        """Bind full slot parameter bundle once for this plugin instance."""
        ...

    def resolve_dataset_plan(self) -> tuple[str, ...]:
        """Resolve canonical dataset iteration order for the host."""
        ...

    def resolve_input_shape(self) -> tuple[int, int]:
        """Resolve canonical input image shape for downstream runtime consumers."""
        ...

    def build_dataset_loaders(
        self,
        *,
        dataset_name: str,
        dataset_idx: int,
    ) -> tuple[Any, Any]:
        """Build train and test loaders for one dataset category."""
        ...

    def build_calibration_train_loader(
        self,
        *,
        dataset_name: str,
    ) -> Any:
        """Build deterministic non-augmented train loader for calibration."""
        ...


__all__ = ["DataLoaderBindContextLike", "DataLoaderPlugin"]
