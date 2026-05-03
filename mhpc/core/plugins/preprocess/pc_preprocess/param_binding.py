"""Plugin-local parameter binding/parsing for `pc_preprocess`."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .config_primitives import _require_int
from ..contracts import PreprocessBindContextLike

_ALLOWED_KEYS = frozenset({"pretrain_embed_dimension"})


@dataclass(frozen=True)
class PCPreprocessParams:
    """Canonical params payload bound to one preprocess plugin instance."""

    pretrain_embed_dimension: int


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


def _require_positive_int(*, value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer.")
    return int(value)


def _parse_preprocess_params(params: Mapping[str, Any]) -> PCPreprocessParams:
    _ensure_allowed_keys(params=params, context="pipeline.slots.preprocess.params")
    pretrain_embed_dimension = _require_positive_int(
        value=_require_int(params, "pretrain_embed_dimension"),
        field_name="pretrain_embed_dimension",
    )
    return PCPreprocessParams(
        pretrain_embed_dimension=pretrain_embed_dimension,
    )


class PCPreprocessParamBindingMixin:
    """Preprocess slot mixin with plugin-local param parsing."""

    _bound_params: dict[str, Any]
    _bound_bind_context: PreprocessBindContextLike
    _preprocess_params: PCPreprocessParams

    def bind_params(
        self,
        *,
        params: Mapping[str, Any],
        bind_context: PreprocessBindContextLike,
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
        self._preprocess_params = _parse_preprocess_params(self._bound_params)
