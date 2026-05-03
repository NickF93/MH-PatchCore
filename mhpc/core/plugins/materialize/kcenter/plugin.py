"""`kcenter` materialization plugin implementation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..common import materialize_global_nn_bank
from ..contracts import (
    MaterializationBindContextLike,
    MaterializationInputState,
    MaterializationPlugin,
    MemoryBankPayload,
)


class KCenterMaterializationPlugin(MaterializationPlugin):
    """Materialization plugin for `kcenter`."""

    supports_streaming: bool = True
    requires_full_dataset: bool = False
    requires_locality_context: bool = False
    preserves_locality: bool = False
    _bound_params: dict[str, Any]
    _bound_bind_context: MaterializationBindContextLike
    _compute_self_distances: bool

    def bind_params(
        self,
        *,
        params: Mapping[str, Any],
        bind_context: MaterializationBindContextLike,
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
        unknown_keys = sorted(
            str(key)
            for key in set(self._bound_params.keys()) - {"compute_self_distances"}
        )
        if unknown_keys:
            raise ValueError(
                "pipeline.slots.materialize.params must be empty except for optional "
                "compute_self_distances for "
                "materialize plugin 'kcenter'; unsupported keys: "
                f"{', '.join(unknown_keys)}"
            )
        raw_compute_self_distances = self._bound_params.get("compute_self_distances", False)
        if not isinstance(raw_compute_self_distances, bool):
            raise TypeError(
                "materialize plugin 'kcenter' requires boolean "
                "params.compute_self_distances."
            )
        self._compute_self_distances = bool(raw_compute_self_distances)

    def materialize(
        self,
        *,
        state: MaterializationInputState,
        locality_context: object | None = None,
    ) -> tuple[MemoryBankPayload, dict[str, object]]:
        del locality_context
        return (
            materialize_global_nn_bank(
                state.get_centroids(),
                compute_self_distances=bool(
                    getattr(self, "_compute_self_distances", False)
                ),
                plugin_name="kcenter",
            ),
            state.export_state(),
        )
