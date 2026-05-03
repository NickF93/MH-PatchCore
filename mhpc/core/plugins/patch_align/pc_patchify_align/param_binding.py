"""Plugin-local parameter binding/parsing for `pc_patchify_align`."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ..contracts import PatchAlignBindContextLike
from .config_primitives import _require_int
from .patch_maker import PatchMaker

_ALLOWED_KEYS = frozenset({"patchsize", "patchstride"})


@dataclass(frozen=True)
class PCPatchifyAlignParams:
    """Canonical params payload bound to one patch-align plugin instance."""

    patchsize: int
    patchstride: int


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


def _parse_patch_align_params(params: Mapping[str, Any]) -> PCPatchifyAlignParams:
    _ensure_allowed_keys(params=params, context="pipeline.slots.patch_align.params")
    patchsize = _require_positive_int(
        value=_require_int(params, "patchsize"),
        field_name="patchsize",
    )
    patchstride = _require_positive_int(
        value=_require_int(params, "patchstride"),
        field_name="patchstride",
    )
    return PCPatchifyAlignParams(
        patchsize=patchsize,
        patchstride=patchstride,
    )


class PCPatchifyAlignParamBindingMixin:
    """Patch-align slot mixin with plugin-local param parsing."""

    _bound_params: dict[str, Any]
    _bound_bind_context: PatchAlignBindContextLike
    _patch_align_params: PCPatchifyAlignParams

    def bind_params(
        self,
        *,
        params: Mapping[str, Any],
        bind_context: PatchAlignBindContextLike,
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
        self._patch_align_params = _parse_patch_align_params(self._bound_params)

    def build_patch_maker(self) -> PatchMaker:
        return PatchMaker(
            patchsize=self._patch_align_params.patchsize,
            stride=self._patch_align_params.patchstride,
        )
