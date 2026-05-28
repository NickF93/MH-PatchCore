"""`kmeans` materialization plugin implementation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from scipy.spatial import distance as scipy_distance  # type: ignore[import-untyped]

from mhpc.core.plugins.locality_state_contract import StructuredGlobalNNBank

from ..contracts import (
    MaterializationBindContextLike,
    MaterializationInputState,
    MaterializationPlugin,
    MemoryBankPayload,
)


class KMeansMaterializationPlugin(MaterializationPlugin):
    """Materialization plugin for `kmeans`."""

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
                "materialize plugin 'kmeans'; unsupported keys: "
                f"{', '.join(unknown_keys)}"
            )
        raw_compute_self_distances = self._bound_params.get("compute_self_distances", False)
        if not isinstance(raw_compute_self_distances, bool):
            raise TypeError(
                "materialize plugin 'kmeans' requires boolean "
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
            _materialize_global_nn_bank(
                state.get_centroids(),
                compute_self_distances=bool(
                    getattr(self, "_compute_self_distances", False)
                ),
            ),
            state.export_state(),
        )


def _normalize_global_ndarray_bank(memory_bank: MemoryBankPayload) -> np.ndarray:
    if not isinstance(memory_bank, np.ndarray):
        raise ValueError(
            "materialize plugin 'kmeans' supports compute_self_distances only for "
            "global ndarray banks."
        )
    bank_np = np.ascontiguousarray(np.asarray(memory_bank, dtype=np.float32))
    if bank_np.ndim != 2:
        raise ValueError(
            "materialize plugin 'kmeans' requires a 2D global ndarray bank to "
            f"compute self_distances; got shape={bank_np.shape}."
        )
    if int(bank_np.shape[0]) < 2 or int(bank_np.shape[1]) <= 0:
        raise ValueError(
            "materialize plugin 'kmeans' requires bank shape [N, D] with N >= 2 "
            f"and D > 0 to compute self_distances; got shape={bank_np.shape}."
        )
    if not np.all(np.isfinite(bank_np)):
        raise ValueError(
            "materialize plugin 'kmeans' requires finite bank values to compute "
            "self_distances."
        )
    return bank_np


def _compute_global_nn_self_distances(memory_bank: MemoryBankPayload) -> np.ndarray:
    bank_np = _normalize_global_ndarray_bank(memory_bank)
    pairwise_distances = np.asarray(
        scipy_distance.cdist(
            np.asarray(bank_np, dtype=np.float64),
            np.asarray(bank_np, dtype=np.float64),
            metric="euclidean",
        ),
        dtype=np.float64,
    )
    np.fill_diagonal(pairwise_distances, np.inf)
    self_distances = np.min(pairwise_distances, axis=1).astype(np.float64, copy=False)
    if not np.all(np.isfinite(self_distances)) or np.any(self_distances < 0.0):
        raise RuntimeError("materialize plugin 'kmeans' produced invalid self_distances.")
    return self_distances


def _materialize_global_nn_bank(
    memory_bank: MemoryBankPayload,
    *,
    compute_self_distances: bool,
) -> MemoryBankPayload:
    if not compute_self_distances:
        return memory_bank
    bank_np = _normalize_global_ndarray_bank(memory_bank)
    return StructuredGlobalNNBank(
        features=bank_np,
        self_distances=_compute_global_nn_self_distances(bank_np),
    )
