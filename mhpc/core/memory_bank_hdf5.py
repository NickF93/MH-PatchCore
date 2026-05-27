"""HDF5 serialization for generic memory-bank contract payloads."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py  # type: ignore[import-untyped]
import numpy as np

from mhpc.core.locality_state_helpers import (
    count_memory_bank_references,
    normalize_memory_bank_payload,
)
from mhpc.core.plugins.locality_state_contract import (
    MemoryBankPayload,
    StructuredGlobalDensityBank,
    StructuredGlobalNNBank,
    StructuredLocalMemoryBank,
)

_SCHEMA = "mhpc.memory_bank_artifact.v1"


def write_memory_bank_hdf5(
    *,
    memory_bank: MemoryBankPayload,
    output_path: Path,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Write a memory-bank payload to a deterministic first-class HDF5 artifact."""
    normalized_bank = normalize_memory_bank_payload(memory_bank)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, "w") as handle:
        handle.attrs["schema"] = _SCHEMA
        handle.attrs["reference_count"] = int(
            count_memory_bank_references(normalized_bank)
        )
        if metadata:
            handle.attrs["metadata_json"] = json.dumps(
                _json_ready(metadata),
                allow_nan=False,
                sort_keys=True,
            )
        _write_memory_bank_payload(handle, normalized_bank)

    return output_path


def _write_memory_bank_payload(
    handle: h5py.File,
    memory_bank: MemoryBankPayload,
) -> None:
    if isinstance(memory_bank, np.ndarray):
        handle.attrs["family"] = "global_ndarray"
        handle.create_dataset("features", data=np.asarray(memory_bank, dtype=np.float32))
        return

    if isinstance(memory_bank, StructuredGlobalNNBank):
        handle.attrs["family"] = "structured_global_nn"
        group = handle.create_group("global_nn")
        group.create_dataset(
            "features",
            data=np.asarray(memory_bank.features, dtype=np.float32),
        )
        if memory_bank.self_distances is not None:
            group.create_dataset(
                "self_distances",
                data=np.asarray(memory_bank.self_distances, dtype=np.float64),
            )
        return

    if isinstance(memory_bank, StructuredGlobalDensityBank):
        handle.attrs["family"] = "structured_global_density"
        group = handle.create_group("global_density")
        group.attrs["model_family"] = memory_bank.model_family
        group.attrs["feature_dim"] = int(memory_bank.feature_dim)
        group.attrs["covariance_type"] = memory_bank.covariance_type
        group.attrs["regularization"] = float(memory_bank.regularization)
        group.attrs["seen_samples"] = int(memory_bank.seen_samples)
        group.attrs["update_count"] = int(memory_bank.update_count)
        group.attrs["is_initialized"] = bool(memory_bank.is_initialized)
        group.create_dataset("component_weights", data=memory_bank.component_weights)
        group.create_dataset("component_means", data=memory_bank.component_means)
        group.create_dataset("component_variances", data=memory_bank.component_variances)
        group.create_dataset(
            "component_effective_counts",
            data=memory_bank.component_effective_counts,
        )
        return

    if isinstance(memory_bank, StructuredLocalMemoryBank):
        handle.attrs["family"] = "structured_local"
        group = handle.create_group("local")
        group.attrs["patch_shape"] = np.asarray(memory_bank.patch_shape, dtype=np.int64)
        group.attrs["flatten_order"] = memory_bank.flatten_order
        positions_group = group.create_group("positions")
        for index, position_bank in enumerate(memory_bank.position_banks):
            position_group = positions_group.create_group(f"{index:06d}")
            position_group.create_dataset(
                "position",
                data=np.asarray(position_bank.position, dtype=np.int64),
            )
            position_group.create_dataset(
                "features",
                data=np.asarray(position_bank.features, dtype=np.float32),
            )
        return

    raise TypeError(f"Unsupported memory-bank payload type: {type(memory_bank).__name__}")


def _json_ready(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {
            str(key): _json_ready(item)
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    raise TypeError(
        "Memory-bank artifact metadata value is not JSON-serializable: "
        f"type={type(value).__name__}"
    )


__all__ = ["write_memory_bank_hdf5"]
