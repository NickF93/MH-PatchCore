"""Plugin-local parameter binding/parsing for `euclidean_nn` distance plugin."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..contracts import DistanceBindContextLike

_ALLOWED_KEYS = frozenset({"k"})


class DistanceParamBindingMixin:
    """Distance slot mixin localized to `euclidean_nn`."""

    _bound_params: dict[str, Any]
    _bound_bind_context: DistanceBindContextLike
    _num_neighbors: int

    def bind_params(
        self,
        *,
        params: Mapping[str, Any],
        bind_context: DistanceBindContextLike,
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
        unknown_keys = sorted(str(key) for key in set(self._bound_params.keys()) - _ALLOWED_KEYS)
        if unknown_keys:
            raise ValueError(
                "pipeline.slots.distance.params contains unsupported keys: "
                f"{', '.join(unknown_keys)}"
            )
        raw_k = self._bound_params.get("k", 1)
        if isinstance(raw_k, bool) or not isinstance(raw_k, int) or raw_k <= 0:
            raise ValueError("k must be a positive integer.")
        self._num_neighbors = int(raw_k)

    def resolve_num_neighbors(self) -> int:
        return int(self._num_neighbors)
