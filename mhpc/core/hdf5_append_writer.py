"""Append-only HDF5 writer for flattened batch-major artifact payloads."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import h5py  # type: ignore[import-untyped]
import numpy as np

_SCHEMA = "mhpc.hdf5_append_writer.v1"
_ARRAYS_GROUP = "arrays"
_INDEX_GROUP = "index"
_DEFAULT_ARRAY_NAME = "value"


class HDF5AppendWriter:
    """Append batch-major named arrays to one row-major HDF5 artifact."""

    def __init__(self, output_path: Path) -> None:
        self.output_path = output_path

    def append(
        self,
        *,
        batch_index: int,
        payload: np.ndarray | Mapping[str, Any],
    ) -> tuple[int, int]:
        """Append one batch payload and return the written row interval."""
        if isinstance(batch_index, bool) or not isinstance(batch_index, int):
            raise TypeError("batch_index must be an integer.")
        if batch_index < 0:
            raise ValueError("batch_index must be >= 0.")

        arrays = _normalize_payload_arrays(payload)
        row_count = _validate_payload_rows(arrays)

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(self.output_path, "a") as handle:
            _initialize_file(handle)
            _validate_or_record_array_names(handle, _canonical_array_names(arrays))
            datasets = _prepare_array_datasets(handle, arrays)
            start_row = _current_row_count(handle)
            end_row = start_row + row_count
            snapshot = _snapshot_dataset_shapes(handle, tuple(datasets))
            try:
                for name, array in arrays.items():
                    _append_array(
                        datasets[name],
                        array=array,
                        start_row=start_row,
                        end_row=end_row,
                    )
                _append_index_rows(
                    handle,
                    start_row=start_row,
                    row_count=row_count,
                    batch_index=batch_index,
                )
            except Exception:
                _restore_dataset_shapes(handle, snapshot)
                raise
        return start_row, end_row


def _normalize_payload_arrays(
    payload: np.ndarray | Mapping[str, Any],
) -> dict[str, np.ndarray]:
    if isinstance(payload, np.ndarray):
        return {_DEFAULT_ARRAY_NAME: _normalize_array(payload, _DEFAULT_ARRAY_NAME)}
    if not isinstance(payload, Mapping):
        raise TypeError("payload must be a numpy array or a mapping of named arrays.")
    if not payload:
        raise ValueError("payload mapping must not be empty.")

    arrays: dict[str, np.ndarray] = {}
    for raw_name, raw_value in payload.items():
        name = str(raw_name)
        if not name:
            raise ValueError("payload array names must be non-empty strings.")
        if name in arrays:
            raise ValueError(f"payload contains duplicate array name: {name}")
        arrays[name] = _normalize_array(raw_value, name)
    return arrays


def _normalize_array(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim == 0:
        raise ValueError(f"payload array '{name}' must include a batch dimension.")
    if int(array.shape[0]) <= 0:
        raise ValueError(f"payload array '{name}' must contain at least one row.")
    return np.ascontiguousarray(array)


def _validate_payload_rows(arrays: Mapping[str, np.ndarray]) -> int:
    row_counts = {name: int(array.shape[0]) for name, array in arrays.items()}
    unique_row_counts = set(row_counts.values())
    if len(unique_row_counts) != 1:
        raise ValueError(
            "payload arrays must share the same batch dimension; "
            f"got {row_counts}."
        )
    return unique_row_counts.pop()


def _initialize_file(handle: h5py.File) -> None:
    if "schema" not in handle.attrs:
        handle.attrs["schema"] = _SCHEMA
    elif handle.attrs["schema"] != _SCHEMA:
        raise ValueError(
            "HDF5 append artifact schema mismatch: "
            f"expected {_SCHEMA}, got {handle.attrs['schema']!r}."
        )
    handle.require_group(_ARRAYS_GROUP)
    index_group = handle.require_group(_INDEX_GROUP)
    _require_index_dataset(index_group, "batch_index", dtype=np.int64)
    _require_index_dataset(index_group, "row_in_batch", dtype=np.int64)


def _canonical_array_names(arrays: Mapping[str, np.ndarray]) -> tuple[str, ...]:
    return tuple(sorted(arrays))


def _validate_or_record_array_names(handle: h5py.File, names: tuple[str, ...]) -> None:
    encoded_names = json.dumps(list(names), sort_keys=True)
    current = handle.attrs.get("array_names_json")
    if current is None:
        handle.attrs["array_names_json"] = encoded_names
        return
    if current != encoded_names:
        raise ValueError(
            "HDF5 append payload names changed across batches: "
            f"expected {current}, got {encoded_names}."
        )


def _current_row_count(handle: h5py.File) -> int:
    return int(handle[_INDEX_GROUP]["batch_index"].shape[0])


def _append_index_rows(
    handle: h5py.File,
    *,
    start_row: int,
    row_count: int,
    batch_index: int,
) -> None:
    index_group = handle[_INDEX_GROUP]
    batch_indices = np.full((row_count,), int(batch_index), dtype=np.int64)
    row_indices = np.arange(row_count, dtype=np.int64)
    _append_1d(index_group["batch_index"], batch_indices, start_row=start_row)
    _append_1d(index_group["row_in_batch"], row_indices, start_row=start_row)


def _append_1d(dataset: h5py.Dataset, values: np.ndarray, *, start_row: int) -> None:
    end_row = start_row + int(values.shape[0])
    dataset.resize((end_row,))
    dataset[start_row:end_row] = values


def _prepare_array_datasets(
    handle: h5py.File,
    arrays: Mapping[str, np.ndarray],
) -> dict[str, h5py.Dataset]:
    arrays_group = handle[_ARRAYS_GROUP]
    datasets: dict[str, h5py.Dataset] = {}
    for name, array in arrays.items():
        if name not in arrays_group:
            datasets[name] = arrays_group.create_dataset(
                name,
                shape=(0, *array.shape[1:]),
                maxshape=(None, *array.shape[1:]),
                chunks=True,
                dtype=array.dtype,
            )
            continue
        dataset = arrays_group[name]
        if dataset.dtype != array.dtype:
            raise ValueError(
                f"HDF5 append dtype mismatch for array '{name}': "
                f"expected {dataset.dtype}, got {array.dtype}."
            )
        if tuple(dataset.shape[1:]) != tuple(array.shape[1:]):
            raise ValueError(
                f"HDF5 append shape mismatch for array '{name}': "
                f"expected tail {tuple(dataset.shape[1:])}, got {tuple(array.shape[1:])}."
            )
        datasets[name] = dataset
    return datasets


def _append_array(
    dataset: h5py.Dataset,
    *,
    array: np.ndarray,
    start_row: int,
    end_row: int,
) -> None:
    dataset.resize((end_row, *dataset.shape[1:]))
    dataset[start_row:end_row] = array


def _snapshot_dataset_shapes(
    handle: h5py.File,
    array_names: tuple[str, ...],
) -> dict[str, tuple[int, ...]]:
    paths = [
        f"{_INDEX_GROUP}/batch_index",
        f"{_INDEX_GROUP}/row_in_batch",
        *(f"{_ARRAYS_GROUP}/{name}" for name in array_names),
    ]
    return {path: tuple(handle[path].shape) for path in paths}


def _restore_dataset_shapes(
    handle: h5py.File,
    snapshot: Mapping[str, tuple[int, ...]],
) -> None:
    for path, shape in snapshot.items():
        handle[path].resize(shape)


def _require_index_dataset(
    group: h5py.Group,
    name: str,
    *,
    dtype: type[np.int64],
) -> h5py.Dataset:
    if name in group:
        dataset = group[name]
        if dataset.dtype != np.dtype(dtype):
            raise ValueError(
                f"HDF5 append index dtype mismatch for '{name}': "
                f"expected {np.dtype(dtype)}, got {dataset.dtype}."
            )
        if dataset.ndim != 1:
            raise ValueError(f"HDF5 append index '{name}' must be 1D.")
        return dataset
    return group.create_dataset(
        name,
        shape=(0,),
        maxshape=(None,),
        chunks=True,
        dtype=dtype,
    )


__all__ = ["HDF5AppendWriter"]
