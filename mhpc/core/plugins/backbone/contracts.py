"""Slot-local contract facade for backbone plugins."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal, Protocol, TypeAlias, overload, runtime_checkable

import torch


@runtime_checkable
class BackboneBindContextLike(Protocol):
    """Protocol-only bind context surface consumed by backbone plugins."""

    training_contract: str
    seed: int


BackboneSpec: TypeAlias = str


@runtime_checkable
class BackboneFeatureExtractor(Protocol):
    """Contract-only protocol for backbone feature-extractor outputs."""

    @overload
    def __call__(
        self,
        x: torch.Tensor,
        return_dict: Literal[True] = True,
    ) -> dict[str, torch.Tensor]:
        ...

    @overload
    def __call__(
        self,
        x: torch.Tensor,
        return_dict: Literal[False],
    ) -> list[torch.Tensor]:
        """Extract configured feature maps from one input batch."""
        ...

    def close(self) -> None:
        """Release extraction resources."""
        ...


@runtime_checkable
class BackbonePlugin(Protocol):
    """Contract-only protocol for backbone loading and feature extractor wiring."""

    def bind_params(
        self,
        *,
        params: Mapping[str, Any],
        bind_context: BackboneBindContextLike,
    ) -> None:
        """Bind full slot parameter bundle once for this plugin instance."""
        ...

    def initialize_backbone_and_layers(
        self,
        *,
        device: torch.device,
    ) -> tuple[torch.nn.Module, list[str]]:
        """Build backbone on device using plugin-bound selection semantics."""
        ...

    def create_feature_extractor(
        self,
        *,
        backbone: torch.nn.Module,
        resolved_embedding_layers: list[str],
    ) -> BackboneFeatureExtractor:
        """Create feature extractor for the resolved backbone/layer configuration."""
        ...


__all__ = [
    "BackboneBindContextLike",
    "BackbonePlugin",
    "BackboneSpec",
    "BackboneFeatureExtractor",
]
