"""Metric utilities for image-level and pixel-level anomaly evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import logging

import numpy as np
from skimage.measure import label  # type: ignore[import-untyped]
from sklearn.metrics import average_precision_score, roc_auc_score  # type: ignore[import-untyped]

from mhpc.util.progress import create_progress_bar, make_progress_postfix

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class BinaryMetricResult:
    """Container for thresholded and ranking-based binary metrics."""

    auroc: float
    ap: float
    f1: float
    precision: float
    recall: float
    accuracy: float
    weighted_accuracy: float
    threshold: float
    tp: int
    fp: int
    fn: int
    tn: int


@dataclass(frozen=True)
class PixelMetricResult:
    """Pixel-level metrics including AUPRO."""

    binary: BinaryMetricResult
    aupro: float


def compute_binary_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold_policy: str,
) -> BinaryMetricResult:
    """Compute binary classification metrics from labels and anomaly scores.

    Args:
        labels: Binary labels with shape ``[N]`` or flattenable equivalent.
        scores: Continuous anomaly scores with same shape as ``labels``.
        threshold_policy: Threshold policy identifier.

    Returns:
        A :class:`BinaryMetricResult`.
    """
    labels_arr = _flatten_binary(labels, name="labels")
    scores_arr = _flatten_scores(scores, name="scores")

    if labels_arr.shape[0] != scores_arr.shape[0]:
        raise ValueError("labels and scores must contain the same number of elements")

    threshold = select_threshold(labels_arr, scores_arr, threshold_policy)
    preds = (scores_arr >= threshold).astype(np.int32)

    tp = int(np.logical_and(preds == 1, labels_arr == 1).sum())
    fp = int(np.logical_and(preds == 1, labels_arr == 0).sum())
    fn = int(np.logical_and(preds == 0, labels_arr == 1).sum())
    tn = int(np.logical_and(preds == 0, labels_arr == 0).sum())

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2.0 * precision * recall, precision + recall)
    accuracy = _safe_div(tp + tn, tp + tn + fp + fn)

    tnr = _safe_div(tn, tn + fp)
    weighted_accuracy = 0.5 * (recall + tnr)

    auroc = _safe_auroc(labels_arr, scores_arr)
    ap = _safe_ap(labels_arr, scores_arr)

    return BinaryMetricResult(
        auroc=auroc,
        ap=ap,
        f1=f1,
        precision=precision,
        recall=recall,
        accuracy=accuracy,
        weighted_accuracy=weighted_accuracy,
        threshold=threshold,
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
    )


def compute_pixel_metrics(
    gt_masks: np.ndarray,
    pred_maps: np.ndarray,
    threshold_policy: str,
    aupro_max_fpr: float,
    aupro_num_thresholds: int,
    compute_aupro_enabled: bool = True,
) -> PixelMetricResult:
    """Compute pixel-level binary metrics and AUPRO.

    Args:
        gt_masks: Ground-truth masks with shape ``[N, H, W]``.
        pred_maps: Predicted anomaly maps with shape ``[N, H, W]``.
        threshold_policy: Threshold policy for thresholded metrics.
        aupro_max_fpr: Maximum false-positive-rate integration limit.
        aupro_num_thresholds: Number of thresholds for AUPRO integration.
        compute_aupro_enabled: Whether to compute AUPRO. If ``False``,
            the function returns ``NaN`` for AUPRO while still computing
            pixel-level binary metrics.

    Returns:
        A :class:`PixelMetricResult`.
    """
    gt_arr = np.asarray(gt_masks)
    pred_arr = np.asarray(pred_maps)

    if gt_arr.shape != pred_arr.shape:
        raise ValueError(
            "gt_masks and pred_maps must have identical shape, "
            f"got {gt_arr.shape} vs {pred_arr.shape}"
        )
    if gt_arr.ndim != 3:
        raise ValueError("gt_masks and pred_maps must have shape [N, H, W]")

    flat_labels = _flatten_binary(gt_arr, name="gt_masks")
    flat_scores = _flatten_scores(pred_arr, name="pred_maps")

    binary_metrics = compute_binary_metrics(
        labels=flat_labels,
        scores=flat_scores,
        threshold_policy=threshold_policy,
    )

    if compute_aupro_enabled:
        aupro = compute_aupro(
            gt_masks=gt_arr,
            pred_maps=pred_arr,
            max_fpr=aupro_max_fpr,
            num_thresholds=aupro_num_thresholds,
        )
    else:
        aupro = float("nan")

    return PixelMetricResult(binary=binary_metrics, aupro=aupro)


def compute_image_aupro(
    labels: np.ndarray,
    scores: np.ndarray,
    max_fpr: float = 0.30,
    num_thresholds: int = 256,
) -> float:
    """Compute image-level AUPRO using singleton-region representation.

    Each image is represented as a ``1x1`` mask where the single pixel denotes
    the image label and score. This provides a consistent AUPRO-style integral
    at image level using the same machinery as pixel-level AUPRO.
    """
    labels_arr = _flatten_binary(labels, name="labels")
    scores_arr = _flatten_scores(scores, name="scores")

    if labels_arr.shape[0] != scores_arr.shape[0]:
        raise ValueError("labels and scores must contain the same number of elements")

    gt_masks = labels_arr.reshape(-1, 1, 1).astype(np.uint8, copy=False)
    pred_maps = scores_arr.reshape(-1, 1, 1)
    return compute_aupro(
        gt_masks=gt_masks,
        pred_maps=pred_maps,
        max_fpr=max_fpr,
        num_thresholds=num_thresholds,
    )


def select_threshold(labels: np.ndarray, scores: np.ndarray, policy: str) -> float:
    """Select a decision threshold using a configured policy."""
    if policy == "fixed_0_5":
        return 0.5
    if policy != "best_f1_per_dataset":
        raise ValueError(f"Unsupported threshold policy: {policy}")

    if labels.size == 0:
        return 0.5

    return _select_threshold_best_f1(labels=labels, scores=scores)


def _select_threshold_best_f1(labels: np.ndarray, scores: np.ndarray) -> float:
    """Compute best-F1 threshold in O(N log N) via one sorted cumulative sweep."""
    n_samples = int(labels.size)
    if n_samples == 0:
        return 0.5

    order = np.argsort(scores, kind="mergesort")[::-1]
    sorted_scores = scores[order]
    sorted_labels = labels[order].astype(np.int64, copy=False)

    if sorted_scores.size == 0:
        return 0.5

    total_positives = int(sorted_labels.sum())

    pred_positive_count = np.arange(1, n_samples + 1, dtype=np.int64)
    true_positive_count = np.cumsum(sorted_labels, dtype=np.int64)
    false_positive_count = pred_positive_count - true_positive_count

    precision = np.divide(
        true_positive_count,
        true_positive_count + false_positive_count,
        out=np.zeros(n_samples, dtype=np.float64),
        where=(true_positive_count + false_positive_count) != 0,
    )
    recall = np.divide(
        true_positive_count,
        total_positives,
        out=np.zeros(n_samples, dtype=np.float64),
        where=total_positives != 0,
    )
    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros(n_samples, dtype=np.float64),
        where=(precision + recall) != 0,
    )

    # Evaluate only unique score boundaries. Each boundary corresponds to
    # threshold == sorted_scores[idx] and predictions over prefix [0:idx+1].
    boundary_mask = np.r_[sorted_scores[:-1] > sorted_scores[1:], True]
    boundary_idx = np.flatnonzero(boundary_mask)

    candidate_scores = sorted_scores[boundary_idx]
    candidate_f1 = f1[boundary_idx]
    if candidate_scores.size == 0:
        return 0.5

    best_f1 = float(candidate_f1.max())
    best_mask = np.isclose(candidate_f1, best_f1, rtol=1e-12, atol=1e-12)

    # Candidate scores are descending; first tie is highest threshold.
    return float(candidate_scores[best_mask][0])


def compute_aupro(
    gt_masks: np.ndarray,
    pred_maps: np.ndarray,
    max_fpr: float = 0.30,
    num_thresholds: int = 256,
) -> float:
    """Compute AUPRO up to a given false-positive-rate bound.

    The implementation follows the standard region-overlap view:
    - For each threshold, compute region-level overlap (PRO) averaged over
      all connected anomaly regions.
    - Compute false positive rate over all normal pixels.
    - Integrate PRO over FPR in ``[0, max_fpr]`` and normalize by ``max_fpr``.
    """
    if not 0.0 < max_fpr <= 1.0:
        raise ValueError("max_fpr must be in (0, 1]")
    if num_thresholds < 8:
        raise ValueError("num_thresholds must be >= 8")

    gt_arr = np.asarray(gt_masks)
    pred_arr = np.asarray(pred_maps)

    if gt_arr.shape != pred_arr.shape:
        raise ValueError(
            "gt_masks and pred_maps must have identical shape, "
            f"got {gt_arr.shape} vs {pred_arr.shape}"
        )
    if gt_arr.ndim != 3:
        raise ValueError("gt_masks and pred_maps must have shape [N, H, W]")

    gt_bool = gt_arr.astype(bool)
    score_arr = pred_arr.astype(np.float64)

    if gt_bool.sum() == 0:
        LOGGER.warning("AUPRO is undefined because no anomalous pixels are present")
        return float("nan")

    region_index_masks: list[tuple[int, np.ndarray]] = []
    for image_idx in range(gt_bool.shape[0]):
        labeled = label(gt_bool[image_idx], connectivity=1)
        n_regions = int(labeled.max())
        for region_id in range(1, n_regions + 1):
            region = labeled == region_id
            if region.any():
                region_index_masks.append((image_idx, region))

    if not region_index_masks:
        LOGGER.warning("AUPRO is undefined because no connected anomaly regions exist")
        return float("nan")

    normal_mask = np.logical_not(gt_bool)
    normal_pixel_count = int(normal_mask.sum())
    if normal_pixel_count == 0:
        LOGGER.warning("AUPRO is undefined because no normal pixels are present")
        return float("nan")

    thresholds = np.linspace(score_arr.max(), score_arr.min(), num_thresholds)

    fpr_values: list[float] = []
    pro_values: list[float] = []

    with create_progress_bar(
        thresholds,
        desc="AUPRO sweep...",
    ) as threshold_iterator:
        for threshold_idx, threshold in enumerate(threshold_iterator, start=1):
            pred_binary = score_arr >= threshold

            false_positives = int(np.logical_and(pred_binary, normal_mask).sum())
            fpr = false_positives / normal_pixel_count

            region_overlaps: list[float] = []
            for image_idx, region_mask in region_index_masks:
                region_size = int(region_mask.sum())
                if region_size == 0:
                    continue
                overlap = int(np.logical_and(pred_binary[image_idx], region_mask).sum())
                region_overlaps.append(overlap / region_size)

            if not region_overlaps:
                continue

            mean_region_overlap = float(np.mean(region_overlaps))
            fpr_values.append(float(fpr))
            pro_values.append(mean_region_overlap)
            threshold_iterator.set_postfix(
                make_progress_postfix(
                    batch=threshold_idx,
                    total=int(thresholds.shape[0]),
                    phase=f"fpr={fpr:.4f},pro={mean_region_overlap:.4f}",
                ),
                refresh=False,
            )

    if not fpr_values:
        LOGGER.warning("AUPRO could not be computed due to empty threshold sweep")
        return float("nan")

    fpr_arr = np.asarray(fpr_values, dtype=np.float64)
    pro_arr = np.asarray(pro_values, dtype=np.float64)

    valid = fpr_arr <= max_fpr
    if not np.any(valid):
        LOGGER.warning(
            "AUPRO could not be computed because all sampled FPR values exceed max_fpr"
        )
        return float("nan")

    fpr_arr = fpr_arr[valid]
    pro_arr = pro_arr[valid]

    sort_idx = np.argsort(fpr_arr)
    fpr_arr = fpr_arr[sort_idx]
    pro_arr = pro_arr[sort_idx]

    # Enforce monotonic envelope before integration.
    pro_arr = np.maximum.accumulate(pro_arr)

    if fpr_arr[0] > 0.0:
        fpr_arr = np.insert(fpr_arr, 0, 0.0)
        pro_arr = np.insert(pro_arr, 0, pro_arr[0])
    if fpr_arr[-1] < max_fpr:
        fpr_arr = np.append(fpr_arr, max_fpr)
        pro_arr = np.append(pro_arr, pro_arr[-1])

    area = np.trapezoid(pro_arr, fpr_arr)
    return float(area / max_fpr)

def _safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def _safe_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    if np.unique(labels).size < 2:
        LOGGER.warning("AUROC is undefined because only one class is present")
        return float("nan")
    try:
        return float(roc_auc_score(labels, scores))
    except ValueError:
        LOGGER.warning("AUROC is undefined because only one class is present")
        return float("nan")


def _safe_ap(labels: np.ndarray, scores: np.ndarray) -> float:
    if np.unique(labels).size < 2:
        LOGGER.warning("Average precision is undefined because only one class is present")
        return float("nan")
    try:
        return float(average_precision_score(labels, scores))
    except ValueError:
        LOGGER.warning("Average precision is undefined because only one class is present")
        return float("nan")


def _flatten_binary(values: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(values).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty")

    # Convert any non-zero value to class 1 for robust mask handling.
    out = (arr > 0).astype(np.int32)
    return out


def _flatten_scores(values: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")
    return arr
