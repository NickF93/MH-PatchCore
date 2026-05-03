"""Concrete plugin class for ``mvtec_dataloader_augment``."""

from __future__ import annotations

from typing import Any

from ..contracts import DataLoaderPlugin
from .facade import (
    build_calibration_train_loader as _build_calibration_train_loader,
    build_dataset_loaders as _build_dataset_loaders,
    resolve_dataset_plan as _resolve_dataset_plan,
)
from .param_binding import MVTecDataLoaderParamBindingMixin


class MVTecDataLoaderAugmentPlugin(MVTecDataLoaderParamBindingMixin, DataLoaderPlugin):
    """Behavior-preserving plugin for MVTec loader wiring."""

    supports_streaming: bool = True
    requires_full_dataset: bool = False

    def resolve_dataset_plan(self) -> tuple[str, ...]:
        return _resolve_dataset_plan(self._dataloader_params)

    def resolve_input_shape(self) -> tuple[int, int]:
        height, width = self._dataloader_params.img_size
        return int(height), int(width)

    def build_dataset_loaders(
        self,
        *,
        dataset_name: str,
        dataset_idx: int,
    ) -> tuple[Any, Any]:
        return _build_dataset_loaders(
            params=self._dataloader_params,
            bind_context=self._bound_bind_context,
            dataset_name=dataset_name,
            dataset_idx=dataset_idx,
        )

    def build_calibration_train_loader(
        self,
        *,
        dataset_name: str,
    ) -> Any:
        return _build_calibration_train_loader(
            params=self._dataloader_params,
            dataset_name=dataset_name,
        )


__all__ = ["MVTecDataLoaderAugmentPlugin"]
