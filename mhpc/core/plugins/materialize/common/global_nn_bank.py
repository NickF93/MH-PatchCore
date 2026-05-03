"""Global NN-bank helpers shared across materialize plugins."""

from __future__ import annotations

import numpy as np
from scipy.spatial import distance as scipy_distance  # type: ignore[import-untyped]

from ..contracts import MemoryBankPayload, StructuredGlobalNNBank


def _normalize_global_ndarray_bank(
    memory_bank: MemoryBankPayload,
    *,
    plugin_name: str,
) -> np.ndarray:
    if not isinstance(memory_bank, np.ndarray):
        raise ValueError(
            f"materialize plugin {plugin_name!r} supports compute_self_distances "
            "only for global ndarray banks."
        )
    bank_np = np.ascontiguousarray(np.asarray(memory_bank, dtype=np.float32))
    if bank_np.ndim != 2:
        raise ValueError(
            f"materialize plugin {plugin_name!r} requires a 2D global ndarray bank "
            f"to compute self_distances; got shape={bank_np.shape}."
        )
    if int(bank_np.shape[0]) < 2 or int(bank_np.shape[1]) <= 0:
        raise ValueError(
            f"materialize plugin {plugin_name!r} requires bank shape [N, D] with "
            f"N >= 2 and D > 0 to compute self_distances; got shape={bank_np.shape}."
        )
    if not np.all(np.isfinite(bank_np)):
        raise ValueError(
            f"materialize plugin {plugin_name!r} requires finite bank values to "
            "compute self_distances."
        )
    return bank_np


def compute_global_nn_self_distances(
    memory_bank: MemoryBankPayload,
    *,
    plugin_name: str,
) -> np.ndarray:
    """Compute nearest-other-member distances for one global ndarray bank."""
    bank_np = _normalize_global_ndarray_bank(
        memory_bank,
        plugin_name=plugin_name,
    )
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
        raise RuntimeError(
            f"materialize plugin {plugin_name!r} produced invalid self_distances."
        )
    return self_distances


def materialize_global_nn_bank(
    memory_bank: MemoryBankPayload,
    *,
    compute_self_distances: bool,
    plugin_name: str,
) -> MemoryBankPayload:
    """Return either the original bank or a structured NN-bank with self-distances."""
    if not compute_self_distances:
        return memory_bank
    bank_np = _normalize_global_ndarray_bank(
        memory_bank,
        plugin_name=plugin_name,
    )
    return StructuredGlobalNNBank(
        features=bank_np,
        self_distances=compute_global_nn_self_distances(
            bank_np,
            plugin_name=plugin_name,
        ),
    )


__all__ = ["compute_global_nn_self_distances", "materialize_global_nn_bank"]
