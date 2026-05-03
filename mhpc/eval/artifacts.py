"""Artifact rendering and persistence for MH-PatchCore evaluation."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

_IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def denormalize_image(image_chw: np.ndarray) -> np.ndarray:
    """Convert a normalized CHW image tensor/array into RGB uint8 HWC format."""
    arr = np.asarray(image_chw)
    if arr.ndim != 3 or arr.shape[0] != 3:
        raise ValueError(
            "image_chw must have shape [3, H, W], "
            f"got {arr.shape}"
        )

    hwc = np.transpose(arr, (1, 2, 0)).astype(np.float32)
    denorm = np.clip((hwc * _IMAGENET_STD) + _IMAGENET_MEAN, 0.0, 1.0)
    return (denorm * 255.0).round().astype(np.uint8)


def save_prediction_artifacts(
    output_dir: Path,
    image_rgb_u8: np.ndarray,
    gt_mask: np.ndarray,
    pred_score_map: np.ndarray,
    pixel_threshold: float,
    overlay_alpha: float,
) -> None:
    """Save image, heatmap, overlays, and binary segmentation artifacts.

    Args:
        output_dir: Target folder for the sample artifact set.
        image_rgb_u8: RGB uint8 image in shape ``[H, W, 3]``.
        gt_mask: Ground truth mask in shape ``[H, W]``.
        pred_score_map: Predicted score map in shape ``[H, W]``.
        pixel_threshold: Threshold for binary predicted segmentation.
        overlay_alpha: Alpha blend factor in ``[0, 1]``.
    """
    if not 0.0 <= overlay_alpha <= 1.0:
        raise ValueError("overlay_alpha must be in [0, 1]")

    image = _validate_rgb_image(image_rgb_u8)
    gt = _validate_mask(gt_mask, name="gt_mask")
    pred_map = _validate_map(pred_score_map, name="pred_score_map")

    if gt.shape != pred_map.shape:
        raise ValueError(
            "gt_mask and pred_score_map must have the same spatial shape, "
            f"got {gt.shape} and {pred_map.shape}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    heatmap_rgb_u8 = _to_heatmap_rgb(pred_map)
    heatmap_overlay = _alpha_blend(image, heatmap_rgb_u8, overlay_alpha)

    binary_mask_u8 = (pred_map >= pixel_threshold).astype(np.uint8) * 255
    seg_overlay = _overlay_binary_mask(
        image=image,
        binary_mask=binary_mask_u8,
        color=np.asarray([255, 0, 0], dtype=np.uint8),
        alpha=overlay_alpha,
    )

    _save_rgb_png(output_dir / "image.png", image)
    _save_gray_png(output_dir / "gt_mask.png", gt.astype(np.uint8) * 255)
    _save_rgb_png(output_dir / "pred_heatmap.png", heatmap_rgb_u8)
    _save_rgb_png(output_dir / "overlay_heatmap.png", heatmap_overlay)
    _save_gray_png(output_dir / "pred_binary_mask.png", binary_mask_u8)
    _save_rgb_png(output_dir / "overlay_segmentation.png", seg_overlay)


def _validate_rgb_image(image_rgb_u8: np.ndarray) -> np.ndarray:
    arr = np.asarray(image_rgb_u8)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(
            "image_rgb_u8 must have shape [H, W, 3], "
            f"got {arr.shape}"
        )
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def _validate_mask(mask: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(mask)
    if arr.ndim != 2:
        raise ValueError(f"{name} must have shape [H, W], got {arr.shape}")
    return (arr > 0).astype(np.uint8)


def _validate_map(score_map: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(score_map, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"{name} must have shape [H, W], got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")
    return arr


def _to_heatmap_rgb(score_map: np.ndarray) -> np.ndarray:
    min_value = float(score_map.min())
    max_value = float(score_map.max())

    if np.isclose(max_value, min_value):
        normalized = np.zeros_like(score_map, dtype=np.float32)
    else:
        normalized = np.asarray(
            (score_map - min_value) / (max_value - min_value),
            dtype=np.float32,
        )

    heat_u8 = (normalized * 255.0).round().astype(np.uint8)
    heat_bgr = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    return cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)


def _alpha_blend(base_rgb: np.ndarray, overlay_rgb: np.ndarray, alpha: float) -> np.ndarray:
    base_f: np.ndarray = base_rgb.astype(np.float32)
    overlay_f: np.ndarray = overlay_rgb.astype(np.float32)
    blended: np.ndarray = ((1.0 - alpha) * base_f) + (alpha * overlay_f)
    return np.clip(blended, 0.0, 255.0).round().astype(np.uint8)


def _overlay_binary_mask(
    image: np.ndarray,
    binary_mask: np.ndarray,
    color: np.ndarray,
    alpha: float,
) -> np.ndarray:
    if binary_mask.ndim != 2:
        raise ValueError("binary_mask must be 2D")
    if image.shape[:2] != binary_mask.shape:
        raise ValueError("image and binary_mask spatial shapes must match")

    overlay: np.ndarray = image.copy().astype(np.float32)
    selection = binary_mask > 0
    overlay[selection] = ((1.0 - alpha) * overlay[selection]) + (alpha * color)
    return np.clip(overlay, 0.0, 255.0).round().astype(np.uint8)


def _save_rgb_png(path: Path, image_rgb_u8: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image_bgr = cv2.cvtColor(image_rgb_u8, cv2.COLOR_RGB2BGR)
    ok = cv2.imwrite(str(path), image_bgr)
    if not ok:
        raise RuntimeError(f"Failed to write image artifact: {path}")


def _save_gray_png(path: Path, image_u8: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), image_u8)
    if not ok:
        raise RuntimeError(f"Failed to write grayscale artifact: {path}")
