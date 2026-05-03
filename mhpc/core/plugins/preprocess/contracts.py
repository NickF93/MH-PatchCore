"""Slot-local contracts for preprocess plugins."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

import torch


@runtime_checkable
class PreprocessBindContextLike(Protocol):
    """Protocol-only bind context surface consumed by preprocess plugins."""

    training_contract: str
    seed: int


@runtime_checkable
class PreprocessPlugin(Protocol):
    """Contract-only protocol for STEP_3 preprocess behavior."""

    def bind_params(
        self,
        *,
        params: Mapping[str, Any],
        bind_context: PreprocessBindContextLike,
    ) -> None:
        """Bind full slot parameter bundle once for this plugin instance."""
        ...

    def forward_embed_preprocess(
        self,
        *,
        features: list[torch.Tensor],
        forward_modules: torch.nn.ModuleDict,
    ) -> torch.Tensor:
        """Run preprocess stage on embedded features."""
        ...

    def create_preprocessing_module(
        self,
        *,
        input_dims: list[int] | tuple[int, ...],
    ) -> torch.nn.Module:
        """Build preprocess runtime module for embedding-layer pooling."""
        ...


__all__ = ["PreprocessBindContextLike", "PreprocessPlugin"]
