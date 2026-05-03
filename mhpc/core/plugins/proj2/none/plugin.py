"""No-op projector-2 plugin implementation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

import numpy as np
import torch

from mhpc.core.plugins.locality_context_contract import LocalityContext
from ..contracts import (
    Projector2BindContextLike,
    Projector2Plugin,
    Projector2TrainContext,
)


class NoneProjector2Plugin(Projector2Plugin):
    """Projector-2 plugin id `none` preserving identity behavior."""

    supports_streaming: bool = True
    requires_full_dataset: bool = False
    requires_fit_state: bool = False
    requires_locality_context: bool = False
    preserves_locality: bool = True
    _bound_params: dict[str, Any]
    _bound_bind_context: Projector2BindContextLike

    def bind_params(
        self,
        *,
        params: Mapping[str, Any],
        bind_context: Projector2BindContextLike,
    ) -> None:
        if not isinstance(params, Mapping):
            raise TypeError(
                "params must be a mapping for plugin bind_params: "
                f"type={type(params).__name__}"
            )
        training_contract = getattr(bind_context, "training_contract", None)
        if not isinstance(training_contract, str):
            raise TypeError(
                "bind_context.training_contract must be a string: "
                f"type={type(training_contract).__name__}"
            )
        normalized_contract = training_contract.strip().upper()
        if normalized_contract not in {"OFFLINE", "STREAMING"}:
            raise ValueError(
                "bind_context.training_contract must be one of "
                "{'OFFLINE', 'STREAMING'}: "
                f"value={training_contract!r}"
            )
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
                "pipeline.slots.proj2.params must be empty for "
                "proj2 plugin 'none'; unsupported keys: "
                f"{', '.join(unknown_keys)}"
            )

    def forward_embed_projector2(
        self,
        *,
        features: torch.Tensor,
        forward_modules: torch.nn.ModuleDict,
        locality_context: LocalityContext | None = None,
    ) -> torch.Tensor:
        _ = forward_modules, locality_context
        return features

    def resolve_train_context(
        self,
        *,
        training_contract: Literal["OFFLINE", "STREAMING"],
        feature_dim: int,
        device: torch.device,
    ) -> Projector2TrainContext:
        if training_contract not in {"OFFLINE", "STREAMING"}:
            raise ValueError(
                "training_contract must be one of {'OFFLINE', 'STREAMING'}: "
                f"value={training_contract!r}"
            )
        if isinstance(feature_dim, bool) or not isinstance(feature_dim, int) or feature_dim <= 0:
            raise ValueError("feature_dim must be a positive integer.")
        return Projector2TrainContext(
            training_contract=training_contract,
            feature_dim=int(feature_dim),
            device=device,
        )

    def train_start(
        self,
        *,
        context: Projector2TrainContext,
    ) -> None:
        _ = context

    def train_update(
        self,
        *,
        batch: np.ndarray,
        locality_context: LocalityContext | None = None,
        update_context: object | None = None,
    ) -> None:
        _ = batch, locality_context, update_context

    def train_finalize(self) -> None:
        return None

    def infer_projector2(
        self,
        *,
        features: np.ndarray,
        stage: str,
        batch_idx: int | None = None,
        locality_context: LocalityContext | None = None,
    ) -> np.ndarray:
        _ = stage, batch_idx, locality_context
        return features

    def state_export(self) -> object | None:
        return None

    def state_load(
        self,
        *,
        state: object | None,
    ) -> None:
        if state is not None:
            raise ValueError("proj2 plugin 'none' does not accept fitted state.")
