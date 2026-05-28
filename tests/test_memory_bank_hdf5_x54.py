from __future__ import annotations

import json
from pathlib import Path

import h5py  # type: ignore[import-untyped]
import numpy as np
import pytest

from mhpc.core.memory_bank_hdf5 import write_memory_bank_hdf5
from mhpc.core.plugins.locality_state_contract import (
    StructuredGlobalDensityBank,
    StructuredGlobalNNBank,
    StructuredLocalMemoryBank,
    StructuredLocalPositionBank,
)


def test_x54_memory_bank_hdf5_writes_global_ndarray(tmp_path: Path) -> None:
    output_path = write_memory_bank_hdf5(
        memory_bank=np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
        output_path=tmp_path / "memory_bank.h5",
        metadata={"dataset": {"name": "bottle"}},
    )

    with h5py.File(output_path, "r") as handle:
        assert handle.attrs["schema"] == "mhpc.memory_bank_artifact.v1"
        assert handle.attrs["family"] == "global_ndarray"
        assert handle.attrs["reference_count"] == 2
        assert json.loads(handle.attrs["metadata_json"]) == {
            "dataset": {"name": "bottle"}
        }
        np.testing.assert_allclose(
            handle["features"][...],
            np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        )


def test_x54_memory_bank_hdf5_writes_structured_global_nn(tmp_path: Path) -> None:
    output_path = write_memory_bank_hdf5(
        memory_bank=StructuredGlobalNNBank(
            features=np.asarray([[1.0, 2.0]], dtype=np.float32),
            self_distances=np.asarray([0.25], dtype=np.float64),
        ),
        output_path=tmp_path / "memory_bank.h5",
    )

    with h5py.File(output_path, "r") as handle:
        assert handle.attrs["family"] == "structured_global_nn"
        np.testing.assert_allclose(handle["global_nn/features"][...], [[1.0, 2.0]])
        np.testing.assert_allclose(handle["global_nn/self_distances"][...], [0.25])


def test_x54_memory_bank_hdf5_writes_structured_global_density(tmp_path: Path) -> None:
    output_path = write_memory_bank_hdf5(
        memory_bank=StructuredGlobalDensityBank(
            model_family="gmm",
            component_weights=np.asarray([0.4, 0.6], dtype=np.float64),
            component_means=np.asarray([[0.0, 1.0], [2.0, 3.0]], dtype=np.float64),
            component_variances=np.asarray([[1.0, 1.5], [2.0, 2.5]], dtype=np.float64),
            component_effective_counts=np.asarray([4.0, 6.0], dtype=np.float64),
            feature_dim=2,
            covariance_type="diag",
            regularization=1.0e-6,
            seen_samples=10,
            update_count=3,
            is_initialized=True,
        ),
        output_path=tmp_path / "memory_bank.h5",
    )

    with h5py.File(output_path, "r") as handle:
        group = handle["global_density"]
        assert handle.attrs["family"] == "structured_global_density"
        assert group.attrs["model_family"] == "gmm"
        assert group.attrs["feature_dim"] == 2
        assert group.attrs["covariance_type"] == "diag"
        assert group.attrs["seen_samples"] == 10
        np.testing.assert_allclose(group["component_weights"][...], [0.4, 0.6])
        np.testing.assert_allclose(
            group["component_means"][...],
            [[0.0, 1.0], [2.0, 3.0]],
        )


def test_x54_memory_bank_hdf5_writes_structured_local(tmp_path: Path) -> None:
    output_path = write_memory_bank_hdf5(
        memory_bank=StructuredLocalMemoryBank(
            patch_shape=(1, 2),
            position_banks=(
                StructuredLocalPositionBank(
                    position=(0, 0),
                    features=np.asarray([[1.0, 2.0]], dtype=np.float32),
                ),
                StructuredLocalPositionBank(
                    position=(0, 1),
                    features=np.asarray([[3.0, 4.0], [5.0, 6.0]], dtype=np.float32),
                ),
            ),
        ),
        output_path=tmp_path / "memory_bank.h5",
    )

    with h5py.File(output_path, "r") as handle:
        assert handle.attrs["family"] == "structured_local"
        assert handle.attrs["reference_count"] == 3
        np.testing.assert_array_equal(handle["local"].attrs["patch_shape"], [1, 2])
        np.testing.assert_array_equal(
            handle["local/positions/000001/position"][...],
            [0, 1],
        )
        np.testing.assert_allclose(
            handle["local/positions/000001/features"][...],
            [[3.0, 4.0], [5.0, 6.0]],
        )


def test_x54_memory_bank_hdf5_rejects_non_json_metadata(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="metadata value is not JSON-serializable"):
        write_memory_bank_hdf5(
            memory_bank=np.asarray([[1.0]], dtype=np.float32),
            output_path=tmp_path / "memory_bank.h5",
            metadata={"bad": object()},
        )
