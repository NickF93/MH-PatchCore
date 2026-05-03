"""Generic host locality helpers shared by fit and predict orchestration."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol, cast

import numpy as np

from .plugins.locality_context_contract import LocalityContext


class _SupportsInferTransform(Protocol):
    def infer_transform(
        self,
        *,
        features: np.ndarray,
        stage: str,
        batch_idx: int | None = None,
        locality_context: LocalityContext | None = None,
    ) -> np.ndarray: ...


class _SupportsInferProjector1(Protocol):
    def infer_projector1(
        self,
        *,
        features: np.ndarray,
        stage: str,
        batch_idx: int | None = None,
        locality_context: LocalityContext | None = None,
    ) -> np.ndarray: ...


class _SupportsInferProjector2(Protocol):
    def infer_projector2(
        self,
        *,
        features: np.ndarray,
        stage: str,
        batch_idx: int | None = None,
        locality_context: LocalityContext | None = None,
    ) -> np.ndarray: ...


def slot_requires_locality_context(*, plugin: object) -> bool:
    """Return whether the selected slot plugin requires locality metadata."""
    return bool(getattr(plugin, "requires_locality_context", False))


def slot_preserves_locality(*, plugin: object) -> bool:
    """Return whether the selected slot plugin preserves locality structure."""
    return bool(getattr(plugin, "preserves_locality", False))


def build_locality_context(
    *,
    batch_size: int,
    patch_shapes: list[list[int]],
) -> LocalityContext:
    """Build canonical row-major locality metadata from patch-align output."""
    if not patch_shapes:
        raise ValueError("Transform spatial context requires at least one patch shape.")
    reference_shape = patch_shapes[0]
    if len(reference_shape) != 2:
        raise ValueError(
            "Transform spatial patch shape must contain exactly two integers; "
            f"got {reference_shape!r}."
        )
    patch_h = int(reference_shape[0])
    patch_w = int(reference_shape[1])
    if patch_h <= 0 or patch_w <= 0:
        raise ValueError(
            "Transform spatial patch shape must contain positive integers; "
            f"got {(patch_h, patch_w)}."
        )
    return LocalityContext(
        batch_size=int(batch_size),
        patch_shape=(patch_h, patch_w),
    )


def build_locality_context_if_required(
    *,
    batch_size: int,
    patch_shapes: list[list[int]],
    plugins: Iterable[object],
) -> LocalityContext | None:
    """Build locality context only when one of the selected plugins needs it."""
    if not any(slot_requires_locality_context(plugin=plugin) for plugin in plugins):
        return None
    return build_locality_context(
        batch_size=batch_size,
        patch_shapes=patch_shapes,
    )


def slot_locality_kwargs(
    *,
    plugin: object,
    locality_context: LocalityContext | None,
) -> dict[str, LocalityContext | None]:
    """Return locality kwargs only for plugins that require them."""
    if not slot_requires_locality_context(plugin=plugin):
        return {}
    return {"locality_context": locality_context}


def infer_transform_with_locality(
    *,
    transform_plugin: object,
    features: np.ndarray,
    stage: str,
    batch_idx: int | None = None,
    locality_context: LocalityContext | None = None,
) -> np.ndarray:
    """Call transform inference with locality metadata only when required."""
    transform_runtime = cast(_SupportsInferTransform, transform_plugin)
    if slot_requires_locality_context(plugin=transform_plugin):
        return transform_runtime.infer_transform(
            features=features,
            stage=stage,
            batch_idx=batch_idx,
            locality_context=locality_context,
        )
    return transform_runtime.infer_transform(
        features=features,
        stage=stage,
        batch_idx=batch_idx,
    )


def infer_projector1_with_locality(
    *,
    projector_plugin: object,
    features: np.ndarray,
    stage: str,
    batch_idx: int | None = None,
    locality_context: LocalityContext | None = None,
) -> np.ndarray:
    """Call proj1 inference with locality metadata only when required."""
    projector_runtime = cast(_SupportsInferProjector1, projector_plugin)
    if slot_requires_locality_context(plugin=projector_plugin):
        return projector_runtime.infer_projector1(
            features=features,
            stage=stage,
            batch_idx=batch_idx,
            locality_context=locality_context,
        )
    return projector_runtime.infer_projector1(
        features=features,
        stage=stage,
        batch_idx=batch_idx,
    )


def infer_projector2_with_locality(
    *,
    projector_plugin: object,
    features: np.ndarray,
    stage: str,
    batch_idx: int | None = None,
    locality_context: LocalityContext | None = None,
) -> np.ndarray:
    """Call proj2 inference with locality metadata only when required."""
    projector_runtime = cast(_SupportsInferProjector2, projector_plugin)
    if slot_requires_locality_context(plugin=projector_plugin):
        return projector_runtime.infer_projector2(
            features=features,
            stage=stage,
            batch_idx=batch_idx,
            locality_context=locality_context,
        )
    return projector_runtime.infer_projector2(
        features=features,
        stage=stage,
        batch_idx=batch_idx,
    )


__all__ = [
    "build_locality_context",
    "build_locality_context_if_required",
    "infer_projector1_with_locality",
    "infer_projector2_with_locality",
    "infer_transform_with_locality",
    "slot_locality_kwargs",
    "slot_preserves_locality",
    "slot_requires_locality_context",
]
