from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import h5py  # type: ignore[import-untyped]
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from mhpc.core.predict_engine import _batch_major_payload
import mhpc.eval.pipeline as pipeline_module
from mhpc.eval.frozen_replay import (
    FrozenReplayBatch,
    fingerprint_checkpoint_state,
    run_frozen_test_eval,
    run_frozen_train_replay,
)
from mhpc.eval.teacher_export import (
    build_teacher_checkpoint_manifest,
    save_teacher_checkpoint,
    save_teacher_memory_bank_artifact,
    teacher_checkpoint_enabled,
    teacher_memory_bank_enabled,
    teacher_replay_enabled,
)


class _Plugins:
    def as_selection_map(self) -> dict[str, str]:
        return {
            "dataloader": "mvtec_dataloader_augment",
            "backbone": "pretrained_backbone",
            "patch_align": "pc_patchify_align",
            "preprocess": "pc_preprocess",
            "feature_agg": "spca",
            "proj1": "none",
            "transform": "cholesky",
            "proj2": "none",
            "mem_agg": "greedy",
            "materialize": "greedy",
            "distance": "euclidean_nn",
            "scoring": "paper_eq7",
        }


class _Model:
    def __init__(self) -> None:
        self.save_calls: list[str] = []
        self._resolved_embedding_layers = ["layer2", "layer3"]
        self._backbone = torch.nn.Identity()
        self.anomaly_scorer = SimpleNamespace(
            detection_features=np.asarray([[1.0, 2.0]], dtype=np.float32),
        )

    def save_to_path(self, save_path: str) -> None:
        self.save_calls.append(save_path)
        output_dir = Path(save_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "patchcore_params.pkl").write_bytes(b"checkpoint")
        (output_dir / "nnscorer_features.npy").write_bytes(b"features")


def _make_config(*, checkpoint_enabled: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        experiment=SimpleNamespace(name="teacher_unit", seed=123),
        training=SimpleNamespace(
            contract="OFFLINE",
            fit_epochs={
                "feature_agg": 2,
                "proj1": 1,
                "transform": 1,
                "proj2": 1,
                "mem_agg": 1,
            },
        ),
        teacher_export=SimpleNamespace(
            checkpoint=SimpleNamespace(enabled=checkpoint_enabled),
            memory_bank=SimpleNamespace(enabled=True),
            replay=SimpleNamespace(enabled=False, slots=()),
        ),
        plugins=_Plugins(),
        slot_params={
            "feature_agg": {"n_components": 16},
            "transform": {"eps": 1.0e-6},
        },
    )


def _make_train_loader() -> DataLoader[Any]:
    dataset = TensorDataset(torch.zeros((3, 3, 8, 8)), torch.zeros((3, 8, 8)))
    dataset.root = Path("datasets/mvtec/bottle")  # type: ignore[attr-defined]
    dataset.split = "train"  # type: ignore[attr-defined]
    return DataLoader(dataset, batch_size=2, shuffle=False)


def _make_replay_loader(sample_groups: tuple[str, ...]) -> DataLoader[Any]:
    samples: list[dict[str, object]] = []
    for sample_index, sample_group in enumerate(sample_groups):
        samples.append(
            {
                "image": torch.full(
                    (3, 2, 2),
                    float(sample_index),
                    dtype=torch.float32,
                ),
                "metadata": {
                    "dataset_name": "bottle",
                    "dataset_root": "datasets/mvtec/bottle",
                    "split": "test",
                    "sample_group": sample_group,
                    "sample_path": f"{sample_group}/{sample_index:03d}.png",
                    "sample_index": sample_index,
                },
            }
        )
    return DataLoader(samples, batch_size=2, shuffle=False)


def test_x54_teacher_checkpoint_enabled_reads_config_surface() -> None:
    assert teacher_checkpoint_enabled(_make_config(checkpoint_enabled=True)) is True
    assert teacher_checkpoint_enabled(_make_config(checkpoint_enabled=False)) is False
    assert teacher_memory_bank_enabled(_make_config()) is True
    assert teacher_replay_enabled(_make_config()) is False


def test_x54_frozen_train_replay_uses_infer_batch_without_mutating_state() -> None:
    loader = DataLoader(
        TensorDataset(torch.arange(4 * 3 * 2 * 2, dtype=torch.float32).reshape(4, 3, 2, 2)),
        batch_size=3,
        shuffle=False,
    )
    observed: list[FrozenReplayBatch] = []

    class _ReplayModel:
        def __init__(self) -> None:
            self.forward_modules = torch.nn.ModuleDict({"identity": torch.nn.Identity()})
            self._backbone = torch.nn.Identity()
            self._stage_owned_state = {
                "feature_agg": {"opaque_state": np.asarray([1.0, 2.0], dtype=np.float32)}
            }
            self.anomaly_scorer = SimpleNamespace(
                detection_features=np.asarray([[3.0, 4.0]], dtype=np.float32)
            )
            self.batch_shapes: list[tuple[int, ...]] = []

        def fit(self, _loader: Any) -> None:
            raise AssertionError("frozen replay must not call fit")

        def infer_batch(self, images: torch.Tensor) -> SimpleNamespace:
            self.batch_shapes.append(tuple(images.shape))
            scores = images.reshape(images.shape[0], -1).sum(dim=1).numpy()
            maps = [np.full((2, 2), float(score), dtype=np.float32) for score in scores]
            return SimpleNamespace(
                image_scores=[float(score) for score in scores],
                pred_maps=maps,
            )

        def infer_batch_with_slot_outputs(
            self,
            images: torch.Tensor,
            *,
            selected_slots: tuple[str, ...],
        ) -> SimpleNamespace:
            prediction = self.infer_batch(images)
            return SimpleNamespace(
                prediction=prediction,
                slot_outputs={
                    slot: np.zeros((int(images.shape[0]), 2), dtype=np.float32)
                    for slot in selected_slots
                },
            )

    config = _make_config()
    config.teacher_export.replay = SimpleNamespace(
        enabled=True,
        slots=("feature_agg", "transform"),
    )
    model = _ReplayModel()
    before = fingerprint_checkpoint_state(model)

    summary = run_frozen_train_replay(
        config=config,  # type: ignore[arg-type]
        model=model,
        dataset_name="bottle",
        train_loader=loader,
        observe_batch=observed.append,
    )

    assert summary.dataset_name == "bottle"
    assert summary.split == "train"
    assert summary.output_name == "replay"
    assert summary.selected_slots == ("feature_agg", "transform")
    assert summary.batch_count == 2
    assert summary.image_count == 4
    assert model.batch_shapes == [(3, 3, 2, 2), (1, 3, 2, 2)]
    assert fingerprint_checkpoint_state(model) == before
    assert [batch.batch_index for batch in observed] == [0, 1]
    assert [len(batch.prediction.image_scores) for batch in observed] == [3, 1]
    assert [tuple(batch.slot_outputs) for batch in observed] == [
        ("feature_agg", "transform"),
        ("feature_agg", "transform"),
    ]


def test_x54_frozen_train_replay_rejects_checkpoint_state_mutation() -> None:
    loader = DataLoader(TensorDataset(torch.zeros((1, 3, 2, 2))), batch_size=1)

    class _MutatingReplayModel:
        def __init__(self) -> None:
            self.forward_modules = torch.nn.ModuleDict()
            self._backbone = torch.nn.Identity()
            self._stage_owned_state = {
                "transform": {"opaque_state": np.asarray([1.0], dtype=np.float32)}
            }
            self.anomaly_scorer = SimpleNamespace(detection_features=None)

        def infer_batch(self, images: torch.Tensor) -> SimpleNamespace:
            self._stage_owned_state["transform"]["opaque_state"] = np.asarray(
                [2.0],
                dtype=np.float32,
            )
            return SimpleNamespace(
                image_scores=[0.0 for _ in range(int(images.shape[0]))],
                pred_maps=[np.zeros((2, 2), dtype=np.float32)],
            )

        def infer_batch_with_slot_outputs(
            self,
            images: torch.Tensor,
            *,
            selected_slots: tuple[str, ...],
        ) -> SimpleNamespace:
            prediction = self.infer_batch(images)
            return SimpleNamespace(
                prediction=prediction,
                slot_outputs={
                    slot: np.zeros((int(images.shape[0]), 1), dtype=np.float32)
                    for slot in selected_slots
                },
            )

    config = _make_config()
    config.teacher_export.replay = SimpleNamespace(enabled=True, slots=("transform",))

    with pytest.raises(RuntimeError, match="mutated checkpoint state"):
        run_frozen_train_replay(
            config=config,  # type: ignore[arg-type]
            model=_MutatingReplayModel(),
            dataset_name="bottle",
            train_loader=loader,
        )


def test_x54_frozen_train_replay_writes_selected_slot_hdf5_artifacts(
    tmp_path: Path,
) -> None:
    loader = _make_replay_loader(("good", "broken_large", "good"))

    class _ArtifactReplayModel:
        _stage_owned_state: dict[str, dict[str, object]] = {}
        anomaly_scorer = SimpleNamespace(detection_features=None)

        def infer_batch_with_slot_outputs(
            self,
            images: torch.Tensor,
            *,
            selected_slots: tuple[str, ...],
        ) -> SimpleNamespace:
            batch_size = int(images.shape[0])
            slot_outputs: dict[str, object] = {}
            if "transform" in selected_slots:
                slot_outputs["transform"] = np.ones(
                    (batch_size, 2),
                    dtype=np.float32,
                )
            if "distance" in selected_slots:
                slot_outputs["distance"] = {
                    "distance_map": np.arange(batch_size * 4, dtype=np.float32).reshape(
                        batch_size,
                        4,
                    )
                }
            if "scoring" in selected_slots:
                slot_outputs["scoring"] = {
                    "heatmap": np.ones((batch_size, 4), dtype=np.float32),
                    "score": np.arange(batch_size, dtype=np.float32),
                }
            return SimpleNamespace(
                prediction=SimpleNamespace(
                    image_scores=[0.0 for _ in range(batch_size)],
                    pred_maps=[np.zeros((2, 2), dtype=np.float32) for _ in range(batch_size)],
                ),
                slot_outputs=slot_outputs,
            )

    config = _make_config()
    config.teacher_export.replay = SimpleNamespace(
        enabled=True,
        slots=("transform", "distance", "scoring"),
    )

    run_frozen_train_replay(
        config=config,  # type: ignore[arg-type]
        model=_ArtifactReplayModel(),
        dataset_name="bottle",
        train_loader=loader,
        artifacts_root=tmp_path / "artifacts",
    )

    transform_path = (
        tmp_path
        / "artifacts"
        / "bottle"
        / "train"
        / "transform"
        / "good"
        / "replay.h5"
    )
    distance_path = (
        tmp_path
        / "artifacts"
        / "bottle"
        / "train"
        / "distance"
        / "good"
        / "replay.h5"
    )
    scoring_path = (
        tmp_path
        / "artifacts"
        / "bottle"
        / "train"
        / "scoring"
        / "good"
        / "replay.h5"
    )
    defect_scoring_path = (
        tmp_path
        / "artifacts"
        / "bottle"
        / "train"
        / "scoring"
        / "broken_large"
        / "replay.h5"
    )
    metadata_path = (
        tmp_path
        / "artifacts"
        / "bottle"
        / "train"
        / "metadata"
        / "good"
        / "metadata.jsonl"
    )
    assert transform_path.exists()
    assert distance_path.exists()
    assert scoring_path.exists()
    assert defect_scoring_path.exists()
    assert metadata_path.exists()
    assert not (tmp_path / "artifacts" / "bottle" / "transform").exists()
    assert not (tmp_path / "artifacts" / "bottle" / "scoring").exists()
    with h5py.File(transform_path, "r") as handle:
        assert handle["arrays/value"].shape == (2, 2)
        np.testing.assert_array_equal(handle["index/batch_index"][...], [0, 1])
        np.testing.assert_array_equal(handle["index/row_in_batch"][...], [0, 0])
    with h5py.File(distance_path, "r") as handle:
        assert handle["arrays/distance_map"].shape == (2, 4)
    with h5py.File(scoring_path, "r") as handle:
        assert handle["arrays/heatmap"].shape == (2, 4)
        assert handle["arrays/score"].shape == (2,)
    metadata_rows = [
        json.loads(line)
        for line in metadata_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [row["metadata"]["sample_group"] for row in metadata_rows] == [
        "good",
        "good",
    ]


def test_x54_replay_slot_payload_normalizer_preserves_image_rows() -> None:
    patch_major = np.arange(2 * 4 * 3, dtype=np.float32).reshape(8, 3)
    normalized = _batch_major_payload(patch_major, batch_size=2)
    assert normalized.shape == (2, 12)
    np.testing.assert_array_equal(normalized[0], patch_major[:4].reshape(-1))
    np.testing.assert_array_equal(normalized[1], patch_major[4:].reshape(-1))

    batch_major = np.zeros((2, 3, 4, 4), dtype=np.float32)
    assert _batch_major_payload(batch_major, batch_size=2).shape == (2, 3, 4, 4)


def test_x54_frozen_replay_rejects_non_batch_major_slot_payload(
    tmp_path: Path,
) -> None:
    loader = _make_replay_loader(("good", "broken_large"))

    class _BadSlotReplayModel:
        _stage_owned_state: dict[str, dict[str, object]] = {}
        anomaly_scorer = SimpleNamespace(detection_features=None)

        def infer_batch_with_slot_outputs(
            self,
            images: torch.Tensor,
            *,
            selected_slots: tuple[str, ...],
        ) -> SimpleNamespace:
            batch_size = int(images.shape[0])
            assert selected_slots == ("transform",)
            return SimpleNamespace(
                prediction=SimpleNamespace(
                    image_scores=[0.0 for _ in range(batch_size)],
                    pred_maps=[
                        np.zeros((2, 2), dtype=np.float32)
                        for _ in range(batch_size)
                    ],
                ),
                slot_outputs={
                    "transform": np.zeros((batch_size + 1, 2), dtype=np.float32),
                },
            )

    config = _make_config()
    config.teacher_export.replay = SimpleNamespace(enabled=True, slots=("transform",))

    with pytest.raises(ValueError, match="first dimension must match batch size"):
        run_frozen_train_replay(
            config=config,  # type: ignore[arg-type]
            model=_BadSlotReplayModel(),
            dataset_name="bottle",
            train_loader=loader,
            artifacts_root=tmp_path / "artifacts",
        )


def test_x54_frozen_train_replay_fails_when_selected_slot_is_missing(
    tmp_path: Path,
) -> None:
    loader = _make_replay_loader(("good",))

    class _MissingSlotReplayModel:
        _stage_owned_state: dict[str, dict[str, object]] = {}
        anomaly_scorer = SimpleNamespace(detection_features=None)

        def infer_batch_with_slot_outputs(
            self,
            images: torch.Tensor,
            *,
            selected_slots: tuple[str, ...],
        ) -> SimpleNamespace:
            del selected_slots
            return SimpleNamespace(
                prediction=SimpleNamespace(
                    image_scores=[0.0 for _ in range(int(images.shape[0]))],
                    pred_maps=[np.zeros((2, 2), dtype=np.float32)],
                ),
                slot_outputs={},
            )

    config = _make_config()
    config.teacher_export.replay = SimpleNamespace(enabled=True, slots=("transform",))

    with pytest.raises(RuntimeError, match="selected slots did not produce"):
        run_frozen_train_replay(
            config=config,  # type: ignore[arg-type]
            model=_MissingSlotReplayModel(),
            dataset_name="bottle",
            train_loader=loader,
            artifacts_root=tmp_path / "artifacts",
        )


def test_x56_frozen_test_eval_writes_test_split_grouped_artifacts(
    tmp_path: Path,
) -> None:
    loader = _make_replay_loader(("good", "broken_large"))

    class _ArtifactEvalModel:
        _stage_owned_state: dict[str, dict[str, object]] = {}
        anomaly_scorer = SimpleNamespace(detection_features=None)

        def infer_batch_with_slot_outputs(
            self,
            images: torch.Tensor,
            *,
            selected_slots: tuple[str, ...],
        ) -> SimpleNamespace:
            batch_size = int(images.shape[0])
            return SimpleNamespace(
                prediction=SimpleNamespace(
                    image_scores=[0.0 for _ in range(batch_size)],
                    pred_maps=[
                        np.zeros((2, 2), dtype=np.float32)
                        for _ in range(batch_size)
                    ],
                ),
                slot_outputs={
                    "transform": np.ones((batch_size, 2), dtype=np.float32),
                    "distance": {
                        "distance_map": np.ones((batch_size, 4), dtype=np.float32)
                    },
                    "scoring": {
                        "heatmap": np.ones((batch_size, 4), dtype=np.float32),
                        "score": np.arange(batch_size, dtype=np.float32),
                    },
                },
            )

    config = _make_config()
    config.teacher_export.replay = SimpleNamespace(
        enabled=True,
        slots=("transform", "distance", "scoring"),
    )

    summary = run_frozen_test_eval(
        config=config,  # type: ignore[arg-type]
        model=_ArtifactEvalModel(),
        dataset_name="bottle",
        test_loader=loader,
        artifacts_root=tmp_path / "artifacts",
    )

    assert summary.split == "test"
    assert summary.output_name == "eval"
    assert (
        tmp_path
        / "artifacts"
        / "bottle"
        / "test"
        / "transform"
        / "good"
        / "eval.h5"
    ).exists()
    assert (
        tmp_path
        / "artifacts"
        / "bottle"
        / "test"
        / "scoring"
        / "broken_large"
        / "eval.h5"
    ).exists()
    assert not (
        tmp_path / "artifacts" / "bottle" / "test" / "scoring" / "good" / "replay.h5"
    ).exists()


def test_x54_frozen_train_replay_disabled_slots_write_no_artifacts(
    tmp_path: Path,
) -> None:
    loader = DataLoader(TensorDataset(torch.zeros((1, 3, 2, 2))), batch_size=1)

    class _NoSlotReplayModel:
        _stage_owned_state: dict[str, dict[str, object]] = {}
        anomaly_scorer = SimpleNamespace(detection_features=None)

        def infer_batch(self, images: torch.Tensor) -> SimpleNamespace:
            return SimpleNamespace(
                image_scores=[0.0 for _ in range(int(images.shape[0]))],
                pred_maps=[np.zeros((2, 2), dtype=np.float32)],
            )

    config = _make_config()
    config.teacher_export.replay = SimpleNamespace(enabled=False, slots=())

    run_frozen_train_replay(
        config=config,  # type: ignore[arg-type]
        model=_NoSlotReplayModel(),
        dataset_name="bottle",
        train_loader=loader,
        artifacts_root=tmp_path / "artifacts",
    )

    assert not (tmp_path / "artifacts" / "bottle").exists()


def test_x54_save_teacher_checkpoint_writes_model_manifest(
    tmp_path: Path,
) -> None:
    config = _make_config()
    model = _Model()
    train_loader = _make_train_loader()

    checkpoint_dir = save_teacher_checkpoint(
        config=config,  # type: ignore[arg-type]
        model=model,
        dataset_name="bottle",
        train_loader=train_loader,
        artifacts_root=tmp_path / "artifacts",
    )

    assert (
        checkpoint_dir
        == tmp_path / "artifacts" / "bottle" / "model" / "checkpoint"
    )
    assert not (tmp_path / "artifacts" / "bottle" / "checkpoint").exists()
    assert model.save_calls == [str(checkpoint_dir)]
    manifest_path = checkpoint_dir / "manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema"] == "mhpc.teacher_checkpoint.v1"
    assert manifest["experiment"] == {"name": "teacher_unit", "seed": 123}
    assert manifest["dataset"]["name"] == "bottle"
    assert manifest["dataset"]["root"] == "datasets/mvtec/bottle"
    assert manifest["dataset"]["split"] == "train"
    assert manifest["dataset"]["sample_count"] == 3
    assert manifest["training"]["fit_epochs"]["feature_agg"] == 2
    assert manifest["plugins"]["backbone"] == "pretrained_backbone"
    assert manifest["slot_params"]["feature_agg"] == {"n_components": 16}
    assert manifest["backbone"] == {
        "module_type": "Identity",
        "resolved_embedding_layers": ["layer2", "layer3"],
        "weights_exported": False,
    }
    assert manifest["checkpoint"]["path"] == str(checkpoint_dir)
    assert manifest["checkpoint"]["model_state_entrypoint"] == "patchcore_params.pkl"
    assert manifest["checkpoint"]["payload_files"] == [
        "nnscorer_features.npy",
        "patchcore_params.pkl",
    ]


def test_x54_teacher_checkpoint_manifest_rejects_non_json_config_values(
    tmp_path: Path,
) -> None:
    config = _make_config()
    config.slot_params["feature_agg"]["bad"] = object()

    with pytest.raises(TypeError, match="Teacher checkpoint manifest value"):
        build_teacher_checkpoint_manifest(
            config=config,  # type: ignore[arg-type]
            model=_Model(),
            dataset_name="bottle",
            train_loader=_make_train_loader(),
            checkpoint_dir=tmp_path,
        )


def test_x54_save_teacher_memory_bank_artifact_writes_model_hdf5(
    tmp_path: Path,
) -> None:
    output_path = save_teacher_memory_bank_artifact(
        config=_make_config(),  # type: ignore[arg-type]
        model=_Model(),
        dataset_name="bottle",
        train_loader=_make_train_loader(),
        artifacts_root=tmp_path / "artifacts",
    )

    assert (
        output_path
        == tmp_path
        / "artifacts"
        / "bottle"
        / "model"
        / "memory_bank"
        / "memory_bank.h5"
    )
    assert not (tmp_path / "artifacts" / "bottle" / "materialize").exists()
    with h5py.File(output_path, "r") as handle:
        assert handle.attrs["schema"] == "mhpc.memory_bank_artifact.v1"
        assert handle.attrs["family"] == "global_ndarray"
        assert handle.attrs["reference_count"] == 1
        np.testing.assert_allclose(handle["features"][...], [[1.0, 2.0]])


def test_x54_run_experiment_saves_teacher_checkpoint_after_fit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    events: list[str] = []
    loader = _make_train_loader()

    class _DataLoaderPlugin:
        def resolve_dataset_plan(self) -> tuple[str, ...]:
            return ("bottle",)

        def build_dataset_loaders(self, **kwargs: Any) -> tuple[Any, Any]:
            assert kwargs["dataset_name"] == "bottle"
            return loader, loader

    class _RuntimeModel:
        def fit(self, train_loader: Any) -> "_RuntimeModel":
            assert train_loader is loader
            events.append("fit")
            return self

        def infer_dataloader(self, test_loader: Any) -> SimpleNamespace:
            assert test_loader is loader
            events.append("infer")
            return SimpleNamespace(
                image_scores=(0.1, 0.9, 0.8),
                image_labels=(0, 1, 1),
                pred_maps=(
                    torch.zeros((8, 8)).numpy(),
                    torch.ones((8, 8)).numpy(),
                    torch.ones((8, 8)).numpy(),
                ),
                gt_masks=(
                    torch.zeros((8, 8)).numpy(),
                    torch.ones((8, 8)).numpy(),
                    torch.ones((8, 8)).numpy(),
                ),
            )

    config = SimpleNamespace(
        experiment=SimpleNamespace(name="teacher_pipeline", seed=7),
        paths=SimpleNamespace(output_root=tmp_path / "output"),
        runtime=SimpleNamespace(device="cpu"),
        training=SimpleNamespace(
            contract="OFFLINE",
            fit_epochs={
                "feature_agg": 1,
                "proj1": 1,
                "transform": 1,
                "proj2": 1,
                "mem_agg": 1,
            },
        ),
        evaluation=SimpleNamespace(
            calibration=SimpleNamespace(
                mode="none",
                eps=1.0e-12,
                apply_to_image=True,
                apply_to_pixel=True,
            ),
            threshold_policy=SimpleNamespace(
                image="fixed_0_5",
                pixel="fixed_0_5",
            ),
            pixel_metrics=SimpleNamespace(
                aupro=SimpleNamespace(
                    max_fpr=0.3,
                    num_thresholds=8,
                    image_enabled=False,
                    pixel_enabled=False,
                ),
            ),
        ),
        artifacts=SimpleNamespace(enabled=False),
        teacher_export=SimpleNamespace(
            checkpoint=SimpleNamespace(enabled=True),
            memory_bank=SimpleNamespace(enabled=True),
            replay=SimpleNamespace(enabled=True, slots=("feature_agg",)),
        ),
        render=SimpleNamespace(
            progress=SimpleNamespace(
                enabled=False,
                leave=False,
                dynamic_ncols=False,
                min_interval=0.0,
            ),
        ),
        plugins=_Plugins(),
        slot_params={"dataloader": {}},
    )

    monkeypatch.setattr(
        pipeline_module,
        "build_runtime_plugin_chain",
        lambda **_kwargs: SimpleNamespace(dataloader_plugin=_DataLoaderPlugin()),
    )
    monkeypatch.setattr(
        pipeline_module,
        "_build_model",
        lambda **_kwargs: _RuntimeModel(),
    )
    monkeypatch.setattr(
        pipeline_module,
        "save_teacher_checkpoint",
        lambda **_kwargs: events.append("checkpoint"),
    )
    monkeypatch.setattr(
        pipeline_module,
        "save_teacher_memory_bank_artifact",
        lambda **_kwargs: events.append("memory_bank"),
    )
    monkeypatch.setattr(
        pipeline_module,
        "run_frozen_train_replay",
        lambda **_kwargs: events.append("replay"),
    )
    monkeypatch.setattr(
        pipeline_module,
        "run_frozen_test_eval",
        lambda **_kwargs: events.append("test_eval"),
    )

    summary = pipeline_module.run_experiment(config)  # type: ignore[arg-type]

    assert events == [
        "fit",
        "checkpoint",
        "memory_bank",
        "replay",
        "infer",
        "test_eval",
    ]
    assert summary["dataset"].tolist() == ["bottle", "MEAN"]
