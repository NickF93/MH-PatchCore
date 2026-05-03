"""Plugin-local parameter binding/parsing for mem_agg plugins."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ..contracts import MemAggBindContextLike

_ALLOWED_KEYS = frozenset({"coreset_percentage"})


@dataclass(frozen=True)
class MemAggParams:
    """Plugin-local params for `greedy`."""

    coreset_percentage: float


def _ensure_allowed_keys(*, params: Mapping[str, Any], context: str) -> None:
    unknown_keys = sorted(str(key) for key in set(params.keys()) - _ALLOWED_KEYS)
    if unknown_keys:
        raise ValueError(f"{context} contains unsupported keys: {', '.join(unknown_keys)}")


def _parse_mem_agg_params(params: Mapping[str, Any]) -> MemAggParams:
    _ensure_allowed_keys(params=params, context="pipeline.slots.mem_agg.params")
    coreset_percentage = float(params.get("coreset_percentage", 0.1))
    if not 0.0 < coreset_percentage <= 1.0:
        raise ValueError("coreset_percentage must be in (0, 1].")
    return MemAggParams(coreset_percentage=coreset_percentage)


class MemoryAggParamBindingMixin:
    """Memory-aggregation param binding localized to each concrete plugin."""

    _bound_params: dict[str, Any]
    _bound_bind_context: MemAggBindContextLike
    _mem_agg_params: MemAggParams

    def bind_params(
        self,
        *,
        params: Mapping[str, Any],
        bind_context: MemAggBindContextLike,
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
        self._mem_agg_params = _parse_mem_agg_params(self._bound_params)
