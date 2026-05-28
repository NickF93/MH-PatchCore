"""Frozen replay orchestration for teacher artifact export."""

from __future__ import annotations

import hashlib
import json
import pickle  # nosec B403
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mhpc.core.batching import NormalizedBatch, normalize_batch
from mhpc.core.hdf5_append_writer import HDF5AppendWriter
from mhpc.core.predict_engine import SlotOutputPayload
from mhpc.eval.config import RunConfig
from mhpc.eval.teacher_export import teacher_replay_slots


@dataclass(frozen=True)
class FrozenReplayBatch:
    """One frozen replay batch observed at a loader boundary."""

    dataset_name: str
    split: str
    batch_index: int
    selected_slots: tuple[str, ...]
    normalized_batch: NormalizedBatch
    prediction: Any
    slot_outputs: Mapping[str, SlotOutputPayload]


@dataclass(frozen=True)
class FrozenReplaySummary:
    """Aggregate accounting for one frozen replay artifact pass."""

    dataset_name: str
    split: str
    output_name: str
    selected_slots: tuple[str, ...]
    batch_count: int
    image_count: int


ReplayBatchObserver = Callable[[FrozenReplayBatch], None]


class SlotReplayArtifactWriter:
    """Append selected frozen replay outputs under the split/sample-group tree."""

    def __init__(
        self,
        *,
        artifacts_root: Path,
        dataset_name: str,
        split: str,
        output_name: str,
        selected_slots: Iterable[str],
    ) -> None:
        self._dataset_root = artifacts_root / dataset_name
        self._split = split
        self._output_name = output_name
        self._selected_slots = tuple(selected_slots)

    def observe(self, replay_batch: FrozenReplayBatch) -> None:
        batch_size = int(replay_batch.normalized_batch.images.shape[0])
        metadata_rows = _metadata_rows(
            replay_batch.normalized_batch.metadata,
            batch_size=batch_size,
        )
        grouped_rows = _group_rows_by_sample_group(metadata_rows)
        for sample_group, row_indices in grouped_rows.items():
            self._append_metadata_rows(
                replay_batch=replay_batch,
                sample_group=sample_group,
                row_indices=row_indices,
                metadata_rows=metadata_rows,
            )

        missing_slots = [
            slot
            for slot in self._selected_slots
            if slot not in replay_batch.slot_outputs
        ]
        if missing_slots:
            raise RuntimeError(
                "Frozen replay selected slots did not produce generic export "
                f"payloads: {', '.join(missing_slots)}."
            )
        for slot_name in self._selected_slots:
            slot_payload = replay_batch.slot_outputs[slot_name]
            _validate_payload_batch_rows(
                slot_payload,
                batch_size=batch_size,
                slot_name=slot_name,
            )
            for sample_group, row_indices in grouped_rows.items():
                output_path = (
                    self._dataset_root
                    / self._split
                    / slot_name
                    / sample_group
                    / f"{self._output_name}.h5"
                )
                HDF5AppendWriter(output_path).append(
                    batch_index=replay_batch.batch_index,
                    payload=_slice_payload_rows(slot_payload, row_indices=row_indices),
                )

    def _append_metadata_rows(
        self,
        *,
        replay_batch: FrozenReplayBatch,
        sample_group: str,
        row_indices: tuple[int, ...],
        metadata_rows: tuple[dict[str, Any], ...],
    ) -> None:
        output_path = (
            self._dataset_root
            / self._split
            / "metadata"
            / sample_group
            / "metadata.jsonl"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("a", encoding="utf-8") as handle:
            for row_index in row_indices:
                record = {
                    "dataset_name": replay_batch.dataset_name,
                    "split": replay_batch.split,
                    "batch_index": int(replay_batch.batch_index),
                    "row_in_batch": int(row_index),
                    "metadata": metadata_rows[row_index],
                }
                json.dump(
                    _json_ready(record, context="metadata"),
                    handle,
                    allow_nan=False,
                    sort_keys=True,
                )
                handle.write("\n")


def run_frozen_train_replay(
    *,
    config: RunConfig,
    model: Any,
    dataset_name: str,
    train_loader: Iterable[Any],
    artifacts_root: Path | None = None,
    observe_batch: ReplayBatchObserver | None = None,
) -> FrozenReplaySummary:
    """Replay the training loader through frozen inference without state mutation."""

    return _run_frozen_replay(
        config=config,
        model=model,
        dataset_name=dataset_name,
        split="train",
        output_name="replay",
        loader=train_loader,
        include_targets=False,
        artifacts_root=artifacts_root,
        observe_batch=observe_batch,
    )


def run_frozen_test_eval(
    *,
    config: RunConfig,
    model: Any,
    dataset_name: str,
    test_loader: Iterable[Any],
    artifacts_root: Path | None = None,
    observe_batch: ReplayBatchObserver | None = None,
) -> FrozenReplaySummary:
    """Replay the test loader through frozen inference for eval artifact export."""

    return _run_frozen_replay(
        config=config,
        model=model,
        dataset_name=dataset_name,
        split="test",
        output_name="eval",
        loader=test_loader,
        include_targets=True,
        artifacts_root=artifacts_root,
        observe_batch=observe_batch,
    )


def _run_frozen_replay(
    *,
    config: RunConfig,
    model: Any,
    dataset_name: str,
    split: str,
    output_name: str,
    loader: Iterable[Any],
    include_targets: bool,
    artifacts_root: Path | None,
    observe_batch: ReplayBatchObserver | None,
) -> FrozenReplaySummary:
    selected_slots = teacher_replay_slots(config)
    artifact_writer = (
        SlotReplayArtifactWriter(
            artifacts_root=artifacts_root,
            dataset_name=dataset_name,
            split=split,
            output_name=output_name,
            selected_slots=selected_slots,
        )
        if artifacts_root is not None and selected_slots
        else None
    )
    before_fingerprint = fingerprint_checkpoint_state(model)
    _set_model_eval_mode(model)

    batch_count = 0
    image_count = 0
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            normalized = normalize_batch(batch, include_targets=include_targets)
            prediction, slot_outputs = _infer_replay_batch(
                model=model,
                normalized=normalized,
                selected_slots=selected_slots,
            )
            batch_size = _validate_prediction_batch_size(
                prediction=prediction,
                expected_batch_size=int(normalized.images.shape[0]),
                dataset_name=dataset_name,
                batch_index=batch_index,
            )
            replay_batch = FrozenReplayBatch(
                dataset_name=dataset_name,
                split=split,
                batch_index=batch_index,
                selected_slots=selected_slots,
                normalized_batch=normalized,
                prediction=prediction,
                slot_outputs=slot_outputs,
            )
            if artifact_writer is not None:
                artifact_writer.observe(replay_batch)
            if observe_batch is not None:
                observe_batch(replay_batch)
            batch_count += 1
            image_count += batch_size

    after_fingerprint = fingerprint_checkpoint_state(model)
    if after_fingerprint != before_fingerprint:
        raise RuntimeError(
            "Frozen replay mutated checkpoint state for "
            f"dataset={dataset_name} split={split}."
        )
    return FrozenReplaySummary(
        dataset_name=dataset_name,
        split=split,
        output_name=output_name,
        selected_slots=selected_slots,
        batch_count=batch_count,
        image_count=image_count,
    )


def _infer_replay_batch(
    *,
    model: Any,
    normalized: NormalizedBatch,
    selected_slots: tuple[str, ...],
) -> tuple[Any, Mapping[str, SlotOutputPayload]]:
    slot_outputs: dict[str, SlotOutputPayload] = {}
    if "dataloader" in selected_slots:
        slot_outputs["dataloader"] = _tensor_payload(normalized.images)

    core_slots = tuple(slot for slot in selected_slots if slot != "dataloader")
    if not core_slots:
        return model.infer_batch(normalized.images), slot_outputs

    infer_with_slots = getattr(model, "infer_batch_with_slot_outputs", None)
    if infer_with_slots is None:
        raise RuntimeError(
            "Frozen replay selected slots require model.infer_batch_with_slot_outputs()."
        )
    replay_output = infer_with_slots(
        normalized.images,
        selected_slots=core_slots,
    )
    prediction = replay_output.prediction
    slot_outputs.update(dict(replay_output.slot_outputs))
    missing_slots = [slot for slot in selected_slots if slot not in slot_outputs]
    if missing_slots:
        raise RuntimeError(
            "Frozen replay selected slots did not produce generic export payloads: "
            f"{', '.join(missing_slots)}."
        )
    return prediction, slot_outputs


def fingerprint_checkpoint_state(model: Any) -> str:
    """Return a deterministic digest of checkpoint-relevant model state."""

    digest = hashlib.sha256()
    _update_fingerprint(
        digest,
        {
            "stage_state": getattr(model, "_stage_owned_state", {}),
            "memory_bank": getattr(
                getattr(model, "anomaly_scorer", None),
                "detection_features",
                None,
            ),
        },
    )
    return digest.hexdigest()


def _set_model_eval_mode(model: Any) -> None:
    for module in (
        getattr(model, "forward_modules", None),
        getattr(model, "_backbone", None),
    ):
        if module is not None and hasattr(module, "eval"):
            module.eval()


def _validate_prediction_batch_size(
    *,
    prediction: Any,
    expected_batch_size: int,
    dataset_name: str,
    batch_index: int,
) -> int:
    image_scores = getattr(prediction, "image_scores", None)
    pred_maps = getattr(prediction, "pred_maps", None)
    if image_scores is None or pred_maps is None:
        raise TypeError(
            "Frozen train replay requires infer_batch() output with "
            "`image_scores` and `pred_maps`."
        )
    score_count = len(image_scores)
    map_count = len(pred_maps)
    if score_count != expected_batch_size or map_count != expected_batch_size:
        raise ValueError(
            "Frozen train replay prediction size mismatch: "
            f"dataset={dataset_name} batch={batch_index} "
            f"expected={expected_batch_size} scores={score_count} maps={map_count}."
        )
    return expected_batch_size


def _tensor_payload(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy()


def _metadata_rows(
    metadata: Mapping[str, Any],
    *,
    batch_size: int,
) -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            str(key): _metadata_value_at_row(
                value,
                row_index=row_index,
                batch_size=batch_size,
            )
            for key, value in metadata.items()
        }
        for row_index in range(batch_size)
    )


def _metadata_value_at_row(
    value: Any,
    *,
    row_index: int,
    batch_size: int,
) -> Any:
    if isinstance(value, torch.Tensor):
        return _metadata_value_at_row(
            value.detach().cpu().numpy(),
            row_index=row_index,
            batch_size=batch_size,
        )
    if isinstance(value, np.ndarray):
        if value.ndim > 0 and int(value.shape[0]) == batch_size:
            return _json_ready(value[row_index], context="metadata")
        return _json_ready(value, context="metadata")
    if isinstance(value, (list, tuple)):
        if len(value) == batch_size:
            return _json_ready(value[row_index], context="metadata")
        return _json_ready(value, context="metadata")
    return _json_ready(value, context="metadata")


def _group_rows_by_sample_group(
    metadata_rows: tuple[dict[str, Any], ...],
) -> dict[str, tuple[int, ...]]:
    grouped: dict[str, list[int]] = {}
    for row_index, metadata in enumerate(metadata_rows):
        sample_group = _validate_sample_group(
            metadata.get("sample_group"),
            row_index=row_index,
        )
        grouped.setdefault(sample_group, []).append(row_index)
    return {
        sample_group: tuple(grouped[sample_group])
        for sample_group in sorted(grouped)
    }


def _validate_sample_group(value: Any, *, row_index: int) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(
            "Frozen replay artifact routing requires metadata.sample_group as "
            f"a non-empty string for row {row_index}."
        )
    group_path = Path(value)
    if (
        group_path.is_absolute()
        or len(group_path.parts) != 1
        or "\\" in value
        or value in {".", ".."}
    ):
        raise ValueError(
            "Frozen replay metadata.sample_group must be one path-safe directory "
            f"name; got {value!r} for row {row_index}."
        )
    return value


def _slice_payload_rows(
    payload: SlotOutputPayload,
    *,
    row_indices: tuple[int, ...],
) -> SlotOutputPayload:
    if isinstance(payload, np.ndarray):
        return np.ascontiguousarray(payload[list(row_indices)])
    if isinstance(payload, Mapping):
        return {
            str(name): np.ascontiguousarray(np.asarray(value)[list(row_indices)])
            for name, value in payload.items()
        }
    raise TypeError(
        "Frozen replay slot payload must be a numpy array or mapping of arrays; "
        f"got {type(payload).__name__}."
    )


def _validate_payload_batch_rows(
    payload: SlotOutputPayload,
    *,
    batch_size: int,
    slot_name: str,
) -> None:
    for array_name, value in _iter_payload_arrays(payload, slot_name=slot_name):
        array = np.asarray(value)
        if array.ndim == 0:
            raise ValueError(
                "Frozen replay slot payload must include a batch dimension: "
                f"slot='{slot_name}' array='{array_name}'."
            )
        if int(array.shape[0]) != batch_size:
            raise ValueError(
                "Frozen replay slot payload first dimension must match batch size: "
                f"slot='{slot_name}' array='{array_name}' "
                f"first_dim={array.shape[0]} batch_size={batch_size}."
            )


def _iter_payload_arrays(
    payload: SlotOutputPayload,
    *,
    slot_name: str,
) -> tuple[tuple[str, Any], ...]:
    if isinstance(payload, np.ndarray):
        return (("value", payload),)
    if isinstance(payload, Mapping):
        return tuple((str(name), value) for name, value in payload.items())
    raise TypeError(
        "Frozen replay slot payload must be a numpy array or mapping of arrays; "
        f"slot='{slot_name}' type={type(payload).__name__}."
    )


def _json_ready(value: Any, *, context: str) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist(), context=context)
    if isinstance(value, torch.Tensor):
        return _json_ready(value.detach().cpu().numpy(), context=context)
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
        "Frozen replay metadata value is not JSON-serializable: "
        f"{context} type={type(value).__name__}"
    )


def _update_fingerprint(digest: Any, value: Any) -> None:
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().contiguous().numpy()
        _update_array_fingerprint(digest, array)
        return
    if isinstance(value, np.ndarray):
        _update_array_fingerprint(digest, value)
        return
    if isinstance(value, Mapping):
        digest.update(b"dict")
        for key in sorted(value, key=str):
            _update_fingerprint(digest, str(key))
            _update_fingerprint(digest, value[key])
        return
    if isinstance(value, tuple):
        digest.update(b"tuple")
        for item in value:
            _update_fingerprint(digest, item)
        return
    if isinstance(value, list):
        digest.update(b"list")
        for item in value:
            _update_fingerprint(digest, item)
        return
    if value is None or isinstance(value, (str, int, float, bool, bytes)):
        digest.update(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))
        return
    try:
        digest.update(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))
    except Exception as exc:  # pragma: no cover - defensive context in error path.
        raise TypeError(
            "Frozen train replay cannot fingerprint checkpoint state value of "
            f"type {type(value).__name__}."
        ) from exc


def _update_array_fingerprint(digest: Any, array: np.ndarray) -> None:
    contiguous = np.ascontiguousarray(array)
    digest.update(b"array")
    digest.update(str(contiguous.dtype).encode("utf-8"))
    digest.update(str(tuple(contiguous.shape)).encode("utf-8"))
    digest.update(contiguous.tobytes())


__all__ = [
    "FrozenReplayBatch",
    "FrozenReplaySummary",
    "ReplayBatchObserver",
    "SlotReplayArtifactWriter",
    "fingerprint_checkpoint_state",
    "run_frozen_test_eval",
    "run_frozen_train_replay",
]
