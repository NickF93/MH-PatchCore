"""No-op transform plugin implementation."""

from __future__ import annotations

from collections.abc import Mapping
import numpy as np
import torch
from typing import Any, Literal, cast

from mhpc.core.plugins.locality_context_contract import LocalityContext
from ..contracts import (
    TransformBindContextLike,
    TransformPlugin,
    TransformRegularizationSettings,
    TransformTrainContext,
)


def _normalize_training_contract(value: object) -> str:
    raw_value = value.name if hasattr(value, "name") else value
    if not isinstance(raw_value, str):
        raise TypeError(
            "training_contract must be a string token: "
            f"type={type(raw_value).__name__}"
        )
    normalized = raw_value.strip().upper()
    if normalized not in {"OFFLINE", "STREAMING"}:
        raise ValueError(
            "training_contract must be one of {'OFFLINE', 'STREAMING'}: "
            f"value={raw_value!r}"
        )
    return normalized


class NoneTransformPlugin(TransformPlugin):
    """Transform plugin id `none` preserving current no-op forward behavior."""

    supports_streaming: bool = True
    requires_full_dataset: bool = False
    requires_fit_state: bool = False
    requires_locality_context: bool = False
    preserves_locality: bool = True
    _bound_params: dict[str, Any]
    _bound_bind_context: TransformBindContextLike

    def bind_params(
        self,
        *,
        params: Mapping[str, Any],
        bind_context: TransformBindContextLike,
    ) -> None:
        if not isinstance(params, Mapping):
            raise TypeError(
                "params must be a mapping for plugin bind_params: "
                f"type={type(params).__name__}"
            )
        _normalize_training_contract(getattr(bind_context, "training_contract", None))
        seed = getattr(bind_context, "seed", None)
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError(
                "bind_context.seed must be an integer: "
                f"type={type(seed).__name__}"
            )
        self._bound_params = dict(params)
        self._bound_bind_context = bind_context
        unknown_keys = sorted(str(key) for key in self._bound_params.keys())
        if unknown_keys:
            raise ValueError(
                "pipeline.slots.transform.params must be empty for "
                "transform plugin 'none'; unsupported keys: "
                f"{', '.join(unknown_keys)}"
            )

    def resolve_train_context(
        self,
        *,
        training_contract: Literal["OFFLINE", "STREAMING"],
        feature_dim: int,
    ) -> TransformTrainContext:
        normalized_contract = _normalize_training_contract(training_contract)
        if feature_dim <= 0:
            raise ValueError("Transform feature_dim must be a positive integer.")
        return TransformTrainContext(
            training_contract=cast(Literal["OFFLINE", "STREAMING"], normalized_contract),
            feature_dim=int(feature_dim),
            regularization=TransformRegularizationSettings(
                enabled=False,
                method="JITTER_ONLY",
                shrinkage="auto",
                eigen_floor_ratio=0.0,
                min_jitter=1.0e-12,
                max_jitter=1.0,
                jitter_multiplier=10.0,
            ),
        )

    def forward_embed_transform(
        self,
        *,
        features: torch.Tensor,
        forward_modules: torch.nn.ModuleDict,
        locality_context: LocalityContext | None = None,
    ) -> torch.Tensor:
        _ = forward_modules, locality_context
        return features

    def train_start(
        self,
        *,
        context: TransformTrainContext,
    ) -> None:
        if context.training_contract not in {"OFFLINE", "STREAMING"}:
            raise ValueError(
                "Transform training contract must be OFFLINE or STREAMING; "
                f"got {context.training_contract!r}."
            )
        if context.feature_dim <= 0:
            raise ValueError("Transform feature_dim must be a positive integer.")

    def train_update(
        self,
        *,
        batch: np.ndarray,
        locality_context: LocalityContext | None = None,
        update_context: object | None = None,
    ) -> None:
        _ = batch, locality_context, update_context
        return

    def train_finalize(self) -> None:
        return

    def infer_transform(
        self,
        *,
        features: np.ndarray,
        stage: str,
        batch_idx: int | None = None,
        locality_context: LocalityContext | None = None,
    ) -> np.ndarray:
        _ = stage, batch_idx, locality_context
        return np.asarray(features)

    def state_export(self) -> object | None:
        return None

    def state_load(
        self,
        *,
        state: object | None,
    ) -> None:
        if state is None:
            return
        if isinstance(state, dict) and not state:
            return
        raise ValueError(
            "Transform 'none' checkpoint state must be null or an empty mapping."
        )
