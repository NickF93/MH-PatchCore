"""Plugin-local parameter binding/parsing for `pretrained_backbone`."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .config_primitives import (
    _require_non_empty_string,
    _require_string_list,
)
from ..contracts import BackboneBindContextLike

_ALLOWED_KEYS = frozenset({"backbone", "embedding_layers"})


@dataclass(frozen=True)
class PretrainedBackboneParams:
    """Canonical params payload bound to one pretrained-backbone instance."""

    backbone: str
    embedding_layers: tuple[str, ...]


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


def _ensure_allowed_keys(*, params: Mapping[str, Any], context: str) -> None:
    unknown_keys = sorted(str(key) for key in set(params.keys()) - _ALLOWED_KEYS)
    if unknown_keys:
        raise ValueError(
            f"{context} contains unsupported keys: {', '.join(unknown_keys)}"
        )


def _parse_pretrained_backbone_params(
    params: Mapping[str, Any],
) -> PretrainedBackboneParams:
    _ensure_allowed_keys(params=params, context="pipeline.slots.backbone.params")
    backbone = _require_non_empty_string(params, "backbone")
    embedding_layers = tuple(_require_string_list(params, "embedding_layers"))
    if not embedding_layers:
        raise ValueError("embedding_layers must not be empty.")
    return PretrainedBackboneParams(
        backbone=backbone,
        embedding_layers=embedding_layers,
    )


class PretrainedBackboneParamBindingMixin:
    """Pretrained-backbone slot mixin with plugin-local param parsing."""

    _bound_params: dict[str, Any]
    _bound_bind_context: BackboneBindContextLike
    _backbone_params: PretrainedBackboneParams

    def bind_params(
        self,
        *,
        params: Mapping[str, Any],
        bind_context: BackboneBindContextLike,
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
        self._backbone_params = _parse_pretrained_backbone_params(self._bound_params)
