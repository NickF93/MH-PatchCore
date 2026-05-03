"""Data loading wiring for experiment execution."""

from __future__ import annotations

from typing import Any, cast

from mhpc.core.plugins.dataloader.contracts import (
    DataLoaderPlugin,
)
def resolve_dataset_plan(
    dataloader_plugin: DataLoaderPlugin,
) -> tuple[str, ...]:
    """Resolve canonical dataset iteration order from the dataloader plugin."""
    dataset_plan = dataloader_plugin.resolve_dataset_plan()
    if not isinstance(dataset_plan, tuple):
        raise TypeError(
            "Dataloader plugin must return a tuple[str, ...] from "
            "resolve_dataset_plan."
        )
    if any(not isinstance(dataset_name, str) or not dataset_name for dataset_name in dataset_plan):
        raise TypeError(
            "Dataloader plugin resolve_dataset_plan must return only non-empty strings."
        )
    return dataset_plan


def build_dataset_loaders(
    *,
    dataset_name: str,
    dataset_idx: int,
    dataloader_plugin: DataLoaderPlugin,
) -> tuple[Any, Any]:
    """Build train and test loaders for one dataset category."""
    loaders = dataloader_plugin.build_dataset_loaders(
        dataset_name=dataset_name,
        dataset_idx=dataset_idx,
    )
    if not isinstance(loaders, tuple) or len(loaders) != 2:
        raise TypeError(
            "Dataloader plugin must return a 2-tuple "
            "(train_loader, test_loader) from build_dataset_loaders; "
            f"dataset='{dataset_name}' type='{type(loaders).__name__}'"
        )
    return cast(tuple[Any, Any], loaders)


def build_calibration_train_loader(
    *,
    dataset_name: str,
    dataloader_plugin: DataLoaderPlugin,
) -> Any:
    """Build a deterministic, non-augmented train loader for calibration fitting."""
    calibration_loader = dataloader_plugin.build_calibration_train_loader(
        dataset_name=dataset_name,
    )
    if calibration_loader is None:
        raise TypeError(
            "Dataloader plugin must return a calibration train loader from "
            "build_calibration_train_loader; "
            f"dataset='{dataset_name}'"
        )
    return calibration_loader
