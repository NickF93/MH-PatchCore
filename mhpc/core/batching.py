"""Typed batch normalization utilities for model fit/predict loops."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class NormalizedBatch:
    """Normalized batch payload with explicit image/target contracts."""

    images: torch.Tensor
    labels: list[int]
    masks: list[np.ndarray]


def normalize_batch(
    batch: Any,
    *,
    include_targets: bool,
) -> NormalizedBatch:
    """Normalize heterogeneous batch payloads to a deterministic structure.

    Supported inputs:
    - ``torch.Tensor`` with shape ``[C,H,W]`` or ``[B,C,H,W]``
    - ``dict`` containing ``"image"`` and optional ``"is_anomaly"``, ``"mask"``
    - ``tuple``/``list`` where index 0 is image tensor and index 1 is mask tensor
      (optional when ``include_targets=False``)
    """
    images: torch.Tensor
    labels: list[int] = []
    masks: list[np.ndarray] = []

    if isinstance(batch, torch.Tensor):
        images = _normalize_image_tensor(batch)
    elif isinstance(batch, dict):
        if "image" not in batch:
            raise ValueError("Dictionary batch must contain key 'image'.")
        images = _normalize_image_tensor(batch["image"])
        if include_targets:
            labels = _to_label_list(batch.get("is_anomaly"))
            masks = _to_mask_list(batch.get("mask"))
            if not labels and masks:
                labels = _infer_labels_from_masks(masks)
    elif isinstance(batch, (tuple, list)):
        if not batch:
            raise ValueError("Tuple/list batch must not be empty.")
        images = _normalize_image_tensor(batch[0])
        if include_targets and len(batch) > 1:
            masks = _to_mask_list(batch[1])
            labels = _infer_labels_from_masks(masks)
    else:
        raise TypeError(
            "Unsupported batch type. Expected Tensor, dict, tuple, or list; "
            f"got {type(batch).__name__}."
        )

    return NormalizedBatch(images=images, labels=labels, masks=masks)


def _normalize_image_tensor(images: Any) -> torch.Tensor:
    """Normalize image tensor to shape ``[B,C,H,W]``."""
    if not isinstance(images, torch.Tensor):
        raise TypeError(
            "Batch image payload must be a torch.Tensor; "
            f"got {type(images).__name__}."
        )
    if images.ndim == 3:
        return images.unsqueeze(0)
    if images.ndim == 4:
        return images
    raise ValueError(
        "Image tensor must have shape [C,H,W] or [B,C,H,W]; "
        f"got shape={tuple(images.shape)}."
    )


def _to_label_list(labels: Any) -> list[int]:
    """Convert optional label payload to a Python integer list."""
    if labels is None:
        return []
    if isinstance(labels, torch.Tensor):
        return labels.detach().cpu().to(torch.int64).reshape(-1).tolist()
    if isinstance(labels, np.ndarray):
        return labels.astype(np.int64).reshape(-1).tolist()
    if isinstance(labels, (list, tuple)):
        return [int(value) for value in labels]
    raise TypeError(
        "Label payload must be Tensor, ndarray, list, tuple, or None; "
        f"got {type(labels).__name__}."
    )


def _to_mask_list(masks: Any) -> list[np.ndarray]:
    """Convert optional mask payload to a list of NumPy arrays."""
    if masks is None:
        return []
    if isinstance(masks, torch.Tensor):
        mask_array = masks.detach().cpu().numpy()
        return [np.asarray(mask, dtype=np.float32) for mask in mask_array]
    if isinstance(masks, np.ndarray):
        return [np.asarray(mask, dtype=np.float32) for mask in masks]
    if isinstance(masks, (list, tuple)):
        converted: list[np.ndarray] = []
        for mask in masks:
            if isinstance(mask, torch.Tensor):
                converted.append(np.asarray(mask.detach().cpu().numpy(), dtype=np.float32))
            else:
                converted.append(np.asarray(mask, dtype=np.float32))
        return converted
    raise TypeError(
        "Mask payload must be Tensor, ndarray, list, tuple, or None; "
        f"got {type(masks).__name__}."
    )


def _infer_labels_from_masks(masks: list[np.ndarray]) -> list[int]:
    """Infer binary anomaly labels from pixelwise masks."""
    labels: list[int] = []
    for mask in masks:
        labels.append(1 if float(np.max(mask)) > 0.0 else 0)
    return labels
