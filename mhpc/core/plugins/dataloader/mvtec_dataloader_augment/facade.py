"""Facade-level orchestration helpers for ``mvtec_dataloader_augment``."""

from __future__ import annotations

from typing import Any

from .core import (
    build_calibration_train_loader as _build_calibration_train_loader,
    build_dataset_loaders as _build_dataset_loaders,
    resolve_dataset_plan as _resolve_dataset_plan,
)
from .param_binding import MVTecDataLoaderParams
from ..contracts import DataLoaderBindContextLike


def resolve_dataset_plan(params: MVTecDataLoaderParams) -> tuple[str, ...]:
    """Delegate dataset-plan resolution to plugin core behavior."""
    return _resolve_dataset_plan(params)


def build_dataset_loaders(
    *,
    params: MVTecDataLoaderParams,
    bind_context: DataLoaderBindContextLike,
    dataset_name: str,
    dataset_idx: int,
) -> tuple[Any, Any]:
    """Delegate dataset loader construction."""
    return _build_dataset_loaders(
        params=params,
        bind_context=bind_context,
        dataset_name=dataset_name,
        dataset_idx=dataset_idx,
    )


def build_calibration_train_loader(
    *,
    params: MVTecDataLoaderParams,
    dataset_name: str,
) -> Any:
    """Delegate calibration loader construction."""
    return _build_calibration_train_loader(
        params=params,
        dataset_name=dataset_name,
    )
