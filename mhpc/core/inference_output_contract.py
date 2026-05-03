"""Typed inference output contracts for core->eval boundaries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class InferenceBatchOutput:
    """Typed output for batch-tensor inference."""

    image_scores: tuple[float, ...]
    pred_maps: tuple[np.ndarray, ...]


@dataclass(frozen=True)
class InferenceDatasetOutput:
    """Typed output for dataloader inference with labels/masks."""

    image_scores: tuple[float, ...]
    pred_maps: tuple[np.ndarray, ...]
    image_labels: tuple[int, ...]
    gt_masks: tuple[np.ndarray, ...]


def _coerce_scores(raw_scores: object, *, surface_name: str) -> tuple[float, ...]:
    if not isinstance(raw_scores, Iterable):
        raise TypeError(f"{surface_name}.image_scores must be iterable.")
    return tuple(float(score) for score in raw_scores)


def _coerce_maps(raw_maps: object, *, surface_name: str) -> tuple[np.ndarray, ...]:
    if not isinstance(raw_maps, Iterable):
        raise TypeError(f"{surface_name}.pred_maps must be iterable.")
    return tuple(np.asarray(mask) for mask in raw_maps)


def _coerce_labels(raw_labels: object, *, surface_name: str) -> tuple[int, ...]:
    if not isinstance(raw_labels, Iterable):
        raise TypeError(f"{surface_name}.image_labels must be iterable.")
    coerced: list[int] = []
    for idx, label in enumerate(raw_labels):
        if isinstance(label, bool) or not isinstance(label, (int, np.integer)):
            raise TypeError(
                f"{surface_name}.image_labels[{idx}] must be an integer; "
                f"got {type(label).__name__}."
            )
        coerced.append(int(label))
    return tuple(coerced)


def as_inference_batch_output(raw_output: object) -> InferenceBatchOutput:
    """Normalize supported predict outputs to ``InferenceBatchOutput``."""

    if isinstance(raw_output, InferenceBatchOutput):
        return raw_output

    if not isinstance(raw_output, tuple) or len(raw_output) != 2:
        raise TypeError(
            "Expected batch inference output as InferenceBatchOutput or "
            "tuple[list[float], list[np.ndarray]]."
        )

    raw_scores, raw_maps = raw_output
    return InferenceBatchOutput(
        image_scores=_coerce_scores(raw_scores, surface_name="batch_output"),
        pred_maps=_coerce_maps(raw_maps, surface_name="batch_output"),
    )


def as_inference_dataset_output(raw_output: object) -> InferenceDatasetOutput:
    """Normalize supported predict outputs to ``InferenceDatasetOutput``."""

    if isinstance(raw_output, InferenceDatasetOutput):
        return raw_output

    if not isinstance(raw_output, tuple) or len(raw_output) != 4:
        raise TypeError(
            "Expected dataloader inference output as InferenceDatasetOutput or "
            "tuple[list[float], list[np.ndarray], list[int], list[np.ndarray]]."
        )

    raw_scores, raw_maps, raw_labels, raw_gt_masks = raw_output
    return InferenceDatasetOutput(
        image_scores=_coerce_scores(raw_scores, surface_name="dataset_output"),
        pred_maps=_coerce_maps(raw_maps, surface_name="dataset_output"),
        image_labels=_coerce_labels(raw_labels, surface_name="dataset_output"),
        gt_masks=_coerce_maps(raw_gt_masks, surface_name="dataset_output"),
    )

