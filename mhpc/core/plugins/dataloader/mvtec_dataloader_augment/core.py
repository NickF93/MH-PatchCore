"""Core behavior for the ``mvtec_dataloader_augment`` plugin."""

from __future__ import annotations

from typing import Any

import torch

from .augmentation_policy import resolve_category_augment_cfg
from .dataset_runtime import create_dataloaders
from .param_binding import MVTecDataLoaderParams
from ..contracts import DataLoaderBindContextLike


def resolve_dataset_plan(params: MVTecDataLoaderParams) -> tuple[str, ...]:
    """Return plugin-owned dataset iteration order."""
    return tuple(params.categories)


def resolve_train_augment_mode(
    *,
    params: MVTecDataLoaderParams,
    bind_context: DataLoaderBindContextLike,
) -> str:
    """Resolve train augmentation mode from bound plugin params."""
    if not params.train_augment_enabled:
        return "none"
    if bind_context.training_contract != "STREAMING":
        return "independent"
    return params.streaming_augmentation_policy


def resolve_train_augment_seed(
    *,
    params: MVTecDataLoaderParams,
    bind_context: DataLoaderBindContextLike,
    dataset_idx: int,
) -> int | None:
    """Resolve deterministic augmentation seed for pass-consistent mode."""
    if dataset_idx < 0:
        raise ValueError(f"dataset_idx must be >= 0; got dataset_idx={dataset_idx}")
    if (
        resolve_train_augment_mode(
            params=params,
            bind_context=bind_context,
        )
        != "pass_consistent"
    ):
        return None
    return int(bind_context.seed + dataset_idx * 1_000_003)


def resolve_train_augment_cfg_for_category(
    *,
    params: MVTecDataLoaderParams,
    dataset_name: str,
) -> dict[str, Any]:
    """Resolve category-level augmentation overrides."""
    return resolve_category_augment_cfg(
        base_cfg=params.train_augment_cfg,
        overrides=params.train_augment_overrides,
        category=dataset_name,
        available_categories=params.categories,
    )


def build_dataset_loaders(
    *,
    params: MVTecDataLoaderParams,
    bind_context: DataLoaderBindContextLike,
    dataset_name: str,
    dataset_idx: int,
) -> tuple[Any, Any]:
    """Build train and test loaders for one dataset category."""
    effective_train_augment_cfg = resolve_train_augment_cfg_for_category(
        params=params,
        dataset_name=dataset_name,
    )
    loader_kwargs = _build_common_loader_kwargs(
        params=params,
        dataset_name=dataset_name,
        augment_cfg=effective_train_augment_cfg,
    )
    loader_kwargs.update(
        {
            "augment": params.train_augment_enabled,
            "augment_mode": resolve_train_augment_mode(
                params=params,
                bind_context=bind_context,
            ),
            "augment_seed": resolve_train_augment_seed(
                params=params,
                bind_context=bind_context,
                dataset_idx=dataset_idx,
            ),
        }
    )
    return create_dataloaders(**loader_kwargs)


def build_calibration_train_loader(
    *,
    params: MVTecDataLoaderParams,
    dataset_name: str,
) -> Any:
    """Build deterministic non-augmented train loader for calibration."""
    effective_train_augment_cfg = resolve_train_augment_cfg_for_category(
        params=params,
        dataset_name=dataset_name,
    )
    loader_kwargs = _build_common_loader_kwargs(
        params=params,
        dataset_name=dataset_name,
        augment_cfg=effective_train_augment_cfg,
    )
    loader_kwargs.update(
        {
            "augment": False,
            "augment_mode": "none",
            "augment_seed": None,
        }
    )
    calibration_train_loader, _ = create_dataloaders(**loader_kwargs)
    return calibration_train_loader


def _build_common_loader_kwargs(
    *,
    params: MVTecDataLoaderParams,
    dataset_name: str,
    augment_cfg: dict[str, Any],
) -> dict[str, Any]:
    return {
        "root": str(params.dataset_root),
        "category": dataset_name,
        "batch_size": params.batch_size,
        "num_workers": params.num_workers,
        "augment_cfg": augment_cfg,
        "augment_seed_devices": params.train_augment_seed_devices,
        "dtype": torch.float32,
        "img_size": params.img_size,
    }
