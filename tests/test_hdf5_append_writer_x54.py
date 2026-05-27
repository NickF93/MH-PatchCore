from __future__ import annotations

from pathlib import Path

import h5py  # type: ignore[import-untyped]
import numpy as np
import pytest

import mhpc.core.hdf5_append_writer as hdf5_writer
from mhpc.core.hdf5_append_writer import HDF5AppendWriter


def _assert_index_rows(
    output_path: Path,
    *,
    batch_index: list[int],
    row_in_batch: list[int],
) -> None:
    with h5py.File(output_path, "r") as handle:
        np.testing.assert_array_equal(handle["index/batch_index"][...], batch_index)
        np.testing.assert_array_equal(handle["index/row_in_batch"][...], row_in_batch)


def test_x54_hdf5_append_writer_appends_batches_and_row_index(tmp_path: Path) -> None:
    output_path = tmp_path / "slot.h5"
    writer = HDF5AppendWriter(output_path)

    assert writer.append(
        batch_index=0,
        payload=np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
    ) == (0, 2)
    assert writer.append(
        batch_index=1,
        payload=np.asarray([[5.0, 6.0]], dtype=np.float32),
    ) == (2, 3)

    with h5py.File(output_path, "r") as handle:
        assert handle.attrs["schema"] == "mhpc.hdf5_append_writer.v1"
        assert handle.attrs["array_names_json"] == '["value"]'
        np.testing.assert_allclose(
            handle["arrays/value"][...],
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        )
        np.testing.assert_array_equal(handle["index/batch_index"][...], [0, 0, 1])
        np.testing.assert_array_equal(handle["index/row_in_batch"][...], [0, 1, 0])


def test_x54_hdf5_append_writer_reopens_and_appends_named_arrays(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "slot.h5"
    HDF5AppendWriter(output_path).append(
        batch_index=3,
        payload={
            "features": np.zeros((2, 3), dtype=np.float32),
            "scores": np.asarray([0.1, 0.2], dtype=np.float64),
        },
    )
    HDF5AppendWriter(output_path).append(
        batch_index=4,
        payload={
            "features": np.ones((1, 3), dtype=np.float32),
            "scores": np.asarray([0.3], dtype=np.float64),
        },
    )

    with h5py.File(output_path, "r") as handle:
        assert handle.attrs["array_names_json"] == '["features", "scores"]'
        assert handle["arrays/features"].shape == (3, 3)
        assert handle["arrays/scores"].shape == (3,)
        np.testing.assert_array_equal(handle["index/batch_index"][...], [3, 3, 4])
        np.testing.assert_array_equal(handle["index/row_in_batch"][...], [0, 1, 0])


def test_x54_hdf5_append_writer_rejects_mismatched_batch_dimensions(
    tmp_path: Path,
) -> None:
    writer = HDF5AppendWriter(tmp_path / "slot.h5")

    with pytest.raises(ValueError, match="same batch dimension"):
        writer.append(
            batch_index=0,
            payload={
                "features": np.zeros((2, 3), dtype=np.float32),
                "scores": np.zeros((3,), dtype=np.float32),
            },
        )


def test_x54_hdf5_append_writer_rejects_dtype_and_shape_drift(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "slot.h5"
    writer = HDF5AppendWriter(output_path)
    writer.append(batch_index=0, payload=np.zeros((2, 3), dtype=np.float32))

    with pytest.raises(ValueError, match="dtype mismatch"):
        writer.append(batch_index=1, payload=np.zeros((1, 3), dtype=np.float64))
    _assert_index_rows(output_path, batch_index=[0, 0], row_in_batch=[0, 1])
    with h5py.File(output_path, "r") as handle:
        assert handle["arrays/value"].shape == (2, 3)

    with pytest.raises(ValueError, match="shape mismatch"):
        writer.append(batch_index=1, payload=np.zeros((1, 4), dtype=np.float32))
    _assert_index_rows(output_path, batch_index=[0, 0], row_in_batch=[0, 1])
    with h5py.File(output_path, "r") as handle:
        assert handle["arrays/value"].shape == (2, 3)


def test_x54_hdf5_append_writer_rejects_name_drift_and_scalar_payload(
    tmp_path: Path,
) -> None:
    writer = HDF5AppendWriter(tmp_path / "slot.h5")
    writer.append(batch_index=0, payload={"features": np.zeros((1, 2), dtype=np.float32)})

    with pytest.raises(ValueError, match="payload names changed"):
        writer.append(batch_index=1, payload={"scores": np.zeros((1,), dtype=np.float32)})

    with pytest.raises(ValueError, match="batch dimension"):
        HDF5AppendWriter(tmp_path / "scalar.h5").append(
            batch_index=0,
            payload=np.asarray(1.0, dtype=np.float32),
        )


def test_x54_hdf5_append_writer_accepts_named_arrays_in_different_order(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "slot.h5"
    writer = HDF5AppendWriter(output_path)
    writer.append(
        batch_index=0,
        payload={
            "scores": np.asarray([0.1], dtype=np.float32),
            "features": np.asarray([[1.0, 2.0]], dtype=np.float32),
        },
    )
    writer.append(
        batch_index=1,
        payload={
            "features": np.asarray([[3.0, 4.0]], dtype=np.float32),
            "scores": np.asarray([0.2], dtype=np.float32),
        },
    )

    with h5py.File(output_path, "r") as handle:
        assert handle.attrs["array_names_json"] == '["features", "scores"]'
        np.testing.assert_allclose(
            handle["arrays/features"][...],
            [[1.0, 2.0], [3.0, 4.0]],
        )
        np.testing.assert_allclose(handle["arrays/scores"][...], [0.1, 0.2])
    _assert_index_rows(output_path, batch_index=[0, 1], row_in_batch=[0, 0])


def test_x54_hdf5_append_writer_rolls_back_partial_array_append_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_path = tmp_path / "slot.h5"
    writer = HDF5AppendWriter(output_path)
    writer.append(
        batch_index=0,
        payload={
            "features": np.zeros((1, 2), dtype=np.float32),
            "scores": np.zeros((1,), dtype=np.float32),
        },
    )

    original_append_array = hdf5_writer._append_array

    def fail_on_scores(
        dataset: h5py.Dataset,
        *,
        array: np.ndarray,
        start_row: int,
        end_row: int,
    ) -> None:
        if dataset.name.endswith("/scores"):
            raise RuntimeError("synthetic array append failure")
        original_append_array(
            dataset,
            array=array,
            start_row=start_row,
            end_row=end_row,
        )

    monkeypatch.setattr(hdf5_writer, "_append_array", fail_on_scores)

    with pytest.raises(RuntimeError, match="synthetic array append failure"):
        writer.append(
            batch_index=1,
            payload={
                "features": np.ones((1, 2), dtype=np.float32),
                "scores": np.ones((1,), dtype=np.float32),
            },
        )

    with h5py.File(output_path, "r") as handle:
        assert handle["arrays/features"].shape == (1, 2)
        assert handle["arrays/scores"].shape == (1,)
    _assert_index_rows(output_path, batch_index=[0], row_in_batch=[0])
