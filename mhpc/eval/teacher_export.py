"""Teacher export helpers for dataset-local fitted-model artifacts."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from mhpc.core.locality_state_helpers import resolve_anomaly_scorer_memory_bank
from mhpc.core.memory_bank_hdf5 import write_memory_bank_hdf5
from mhpc.eval.config import RunConfig

_MANIFEST_NAME = "manifest.json"
_MANIFEST_SCHEMA = "mhpc.teacher_checkpoint.v1"


def teacher_checkpoint_enabled(config: RunConfig) -> bool:
    """Return whether the config enables dataset-local teacher checkpoints."""
    return bool(config.teacher_export.checkpoint.enabled)


def teacher_memory_bank_enabled(config: RunConfig) -> bool:
    """Return whether the config enables dataset-local memory-bank artifacts."""
    return bool(config.teacher_export.memory_bank.enabled)


def teacher_replay_enabled(config: RunConfig) -> bool:
    """Return whether frozen train-replay artifact export is enabled."""

    replay_cfg = getattr(config.teacher_export, "replay", None)
    if replay_cfg is None:
        return False
    return bool(replay_cfg.enabled)


def teacher_replay_slots(config: RunConfig) -> tuple[str, ...]:
    """Return configured frozen train-replay slot names."""

    replay_cfg = getattr(config.teacher_export, "replay", None)
    if replay_cfg is None:
        return ()
    return tuple(replay_cfg.slots)


def save_teacher_checkpoint(
    *,
    config: RunConfig,
    model: Any,
    dataset_name: str,
    train_loader: Any,
    artifacts_root: Path,
) -> Path:
    """Persist one fitted teacher checkpoint under artifacts/<dataset>/model/checkpoint."""
    checkpoint_dir = artifacts_root / dataset_name / "model" / "checkpoint"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model.save_to_path(str(checkpoint_dir))

    manifest = build_teacher_checkpoint_manifest(
        config=config,
        model=model,
        dataset_name=dataset_name,
        train_loader=train_loader,
        checkpoint_dir=checkpoint_dir,
    )
    manifest_path = checkpoint_dir / _MANIFEST_NAME
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, allow_nan=False, indent=2, sort_keys=True)
        handle.write("\n")

    return checkpoint_dir


def save_teacher_memory_bank_artifact(
    *,
    config: RunConfig,
    model: Any,
    dataset_name: str,
    train_loader: Any,
    artifacts_root: Path,
) -> Path:
    """Persist the fitted memory bank under artifacts/<dataset>/model/memory_bank."""
    memory_bank = resolve_anomaly_scorer_memory_bank(
        anomaly_scorer=model.anomaly_scorer,
        stage="memory bank artifact export",
    )
    output_path = (
        artifacts_root / dataset_name / "model" / "memory_bank" / "memory_bank.h5"
    )
    metadata = {
        "experiment": {
            "name": config.experiment.name,
            "seed": int(config.experiment.seed),
        },
        "dataset": _collect_dataset_metadata(
            dataset_name=dataset_name,
            train_loader=train_loader,
        ),
        "training": {
            "contract": config.training.contract,
            "fit_epochs": dict(config.training.fit_epochs),
        },
    }
    return write_memory_bank_hdf5(
        memory_bank=memory_bank,
        output_path=output_path,
        metadata=metadata,
    )


def build_teacher_checkpoint_manifest(
    *,
    config: RunConfig,
    model: Any,
    dataset_name: str,
    train_loader: Any,
    checkpoint_dir: Path,
) -> dict[str, Any]:
    """Build a deterministic JSON manifest for a saved teacher checkpoint."""
    payload_files = sorted(
        str(path.relative_to(checkpoint_dir))
        for path in checkpoint_dir.rglob("*")
        if path.is_file() and path.name != _MANIFEST_NAME
    )
    manifest = {
        "schema": _MANIFEST_SCHEMA,
        "experiment": {
            "name": config.experiment.name,
            "seed": int(config.experiment.seed),
        },
        "dataset": _collect_dataset_metadata(
            dataset_name=dataset_name,
            train_loader=train_loader,
        ),
        "training": {
            "contract": config.training.contract,
            "fit_epochs": dict(config.training.fit_epochs),
        },
        "plugins": config.plugins.as_selection_map(),
        "slot_params": config.slot_params,
        "backbone": _collect_backbone_metadata(model=model),
        "checkpoint": {
            "path": str(checkpoint_dir),
            "payload_files": payload_files,
            "model_state_entrypoint": "patchcore_params.pkl",
        },
    }
    return _json_ready(manifest, context="manifest")


def _collect_dataset_metadata(
    *,
    dataset_name: str,
    train_loader: Any,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {"name": dataset_name}
    dataset = getattr(train_loader, "dataset", None)
    if dataset is None:
        return metadata

    metadata["dataset_type"] = type(dataset).__name__
    try:
        metadata["sample_count"] = len(dataset)
    except TypeError:
        pass

    for attr_name in ("root", "dataset_root", "path", "data_path", "split"):
        if hasattr(dataset, attr_name):
            metadata[attr_name] = getattr(dataset, attr_name)

    return metadata


def _collect_backbone_metadata(*, model: Any) -> dict[str, Any]:
    resolved_layers = getattr(model, "_resolved_embedding_layers", ())
    backbone = getattr(model, "_backbone", None)
    return {
        "weights_exported": False,
        "resolved_embedding_layers": list(resolved_layers),
        "module_type": None if backbone is None else type(backbone).__name__,
    }


def _json_ready(value: Any, *, context: str) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {
            str(key): _json_ready(item, context=f"{context}.{key}")
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [
            _json_ready(item, context=f"{context}[{index}]")
            for index, item in enumerate(value)
        ]
    raise TypeError(
        "Teacher checkpoint manifest value is not JSON-serializable: "
        f"{context} type={type(value).__name__}"
    )


__all__ = [
    "build_teacher_checkpoint_manifest",
    "save_teacher_checkpoint",
    "save_teacher_memory_bank_artifact",
    "teacher_checkpoint_enabled",
    "teacher_memory_bank_enabled",
]
