"""Slot-local contracts for patch-align plugins."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

import torch


@runtime_checkable
class PatchAlignBindContextLike(Protocol):
    """Protocol-only bind context surface consumed by patch-align plugins."""

    training_contract: str
    seed: int


@runtime_checkable
class PatchAlignPlugin(Protocol):
    """Contract-only protocol for patchify+align behavior."""

    def bind_params(
        self,
        *,
        params: Mapping[str, Any],
        bind_context: PatchAlignBindContextLike,
    ) -> None:
        """Bind full slot parameter bundle once for this plugin instance."""
        ...

    def create_patch_maker(self) -> Any:
        """Create the canonical patch-maker runtime object for this plugin."""
        ...

    def patchify_and_align(
        self,
        *,
        features: list[torch.Tensor],
        patch_maker: Any,
    ) -> tuple[list[torch.Tensor], list[list[int]]]:
        """Patchify and align multi-layer features to a common patch grid."""
        ...


__all__ = ["PatchAlignBindContextLike", "PatchAlignPlugin"]
