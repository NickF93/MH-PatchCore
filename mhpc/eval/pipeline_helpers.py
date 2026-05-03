"""Helper functions for experiment pipeline orchestration."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import torch

from mhpc.core.plugins.dataloader.contracts import DataLoaderPlugin
from mhpc.core.mh_patch_core import MHPatchCore
from mhpc.core.runtime_plugin_bundle_contract import RuntimePluginBundle
from mhpc.eval.calibration import (
    CalibrationConfig as RuntimeCalibrationConfig,
    ScoreCalibrator,
    build_score_calibrator,
)
from mhpc.eval.config import RunConfig
from mhpc.eval.data_loading import (
    build_calibration_train_loader,
)
from mhpc.eval.profiling import DatasetProfiler
from mhpc.util.progress import (
    artifacts_progress_desc,
    calibration_progress_desc,
    create_progress_bar,
    make_progress_postfix,
)

LOGGER = logging.getLogger(__name__)

def _normalized_training_contract(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError(
            "pipeline.training.contract must be a string: "
            f"type={type(value).__name__}"
        )
    return value.strip().upper()


def _make_run_timestamp() -> str:
    """Create a filesystem-safe local timestamp for one experiment execution."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _validate_strict_streaming_pipeline_contract(config: RunConfig) -> None:
    """Validate strict-streaming constraints for STREAMING experiment runs."""
    training_contract = _normalized_training_contract(config.training.contract)
    if training_contract != "STREAMING":
        return
    if config.evaluation.calibration.mode not in {"none", "zscore"}:
        raise ValueError(
            "STREAMING pipeline strict contract allows evaluation.calibration.mode only "
            "in {'none', 'zscore'}; mode='ecdf' is rejected because ECDF fitting stores "
            "full train prediction arrays."
        )


def _resolve_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_model_impl(
    config: RunConfig,
    device: torch.device,
    *,
    model_cls: type[MHPatchCore],
    plugin_bundle: RuntimePluginBundle | None = None,
) -> MHPatchCore:
    return model_cls(
        device=device,
        training_contract=config.training.contract,
        fit_epochs={
            stage_name: int(value)
            for stage_name, value in config.training.fit_epochs.items()
        },
        plugin_bundle=plugin_bundle,
    )


def _fit_score_calibrator(
    config: RunConfig,
    model: MHPatchCore,
    train_loader: Any,
    dataset_name: str,
) -> ScoreCalibrator | None:
    calibration_cfg = config.evaluation.calibration
    runtime_cfg = RuntimeCalibrationConfig(
        mode=calibration_cfg.mode,
        eps=calibration_cfg.eps,
        apply_to_image=calibration_cfg.apply_to_image,
        apply_to_pixel=calibration_cfg.apply_to_pixel,
    )
    calibrator = build_score_calibrator(runtime_cfg)
    if calibrator is None:
        return None

    if (
        _normalized_training_contract(config.training.contract) == "STREAMING"
        and calibration_cfg.mode == "zscore"
    ):
        return _fit_streaming_zscore_calibrator(
            calibrator=calibrator,
            model=model,
            train_loader=train_loader,
            dataset_name=dataset_name,
        )

    LOGGER.info(
        "Fitting score calibrator for dataset=%s mode=%s",
        dataset_name,
        calibration_cfg.mode,
    )
    train_prediction = model.infer_dataloader(train_loader)
    train_scores_arr = np.asarray(train_prediction.image_scores, dtype=np.float64)
    train_pred_maps_arr = np.asarray(train_prediction.pred_maps, dtype=np.float64)
    if train_scores_arr.size == 0:
        raise RuntimeError(
            f"Cannot fit score calibrator for dataset={dataset_name}: empty train scores."
        )
    if train_pred_maps_arr.size == 0:
        raise RuntimeError(
            f"Cannot fit score calibrator for dataset={dataset_name}: empty train maps."
        )
    calibrator.fit(train_scores_arr, train_pred_maps_arr)
    return calibrator


def _fit_streaming_zscore_calibrator(
    calibrator: ScoreCalibrator,
    model: MHPatchCore,
    train_loader: Any,
    dataset_name: str,
) -> ScoreCalibrator:
    """Fit z-score calibration in streaming mode without global prediction buffers."""
    LOGGER.info(
        "Fitting streaming zscore calibrator for dataset=%s",
        dataset_name,
    )
    processed_batches = 0
    processed_images = 0
    with create_progress_bar(
        train_loader,
        desc=calibration_progress_desc(dataset_name),
    ) as calibration_iterator:
        for batch_idx, batch in enumerate(calibration_iterator, start=1):
            images = _extract_images(batch)
            prediction = model.infer_batch(images)

            batch_scores_arr = np.asarray(prediction.image_scores, dtype=np.float64)
            batch_maps_arr = np.asarray(prediction.pred_maps, dtype=np.float64)
            if batch_scores_arr.size == 0:
                raise RuntimeError(
                    "Cannot fit streaming zscore calibrator: encountered empty "
                    f"scores at dataset={dataset_name}, batch={batch_idx}."
                )
            if batch_maps_arr.size == 0:
                raise RuntimeError(
                    "Cannot fit streaming zscore calibrator: encountered empty "
                    f"score maps at dataset={dataset_name}, batch={batch_idx}."
                )
            calibrator.update(
                image_scores=batch_scores_arr,
                pixel_maps=batch_maps_arr,
            )
            processed_batches += 1
            processed_images += int(batch_scores_arr.shape[0])
            calibration_iterator.set_postfix(
                make_progress_postfix(
                    batch=batch_idx,
                    batch_size=int(batch_scores_arr.shape[0]),
                    images=processed_images,
                    phase="calibration",
                ),
                refresh=False,
            )

    if processed_batches == 0:
        raise RuntimeError(
            f"Cannot fit streaming zscore calibrator for dataset={dataset_name}: "
            "empty train loader."
        )

    calibrator.finalize_fit()
    return calibrator


def _build_calibration_train_loader(
    config: RunConfig,
    dataset_name: str,
    dataloader_plugin: DataLoaderPlugin,
) -> Any:
    """Build a deterministic, non-augmented train loader for calibration fitting."""
    del config
    return build_calibration_train_loader(
        dataset_name=dataset_name,
        dataloader_plugin=dataloader_plugin,
    )


def _validate_prediction_shapes(
    image_scores: np.ndarray,
    image_labels: np.ndarray,
    pred_maps: np.ndarray,
    gt_masks: np.ndarray,
) -> None:
    if image_scores.ndim != 1:
        raise ValueError(f"Expected image_scores to be 1D, got {image_scores.shape}")
    if image_labels.ndim != 1:
        raise ValueError(f"Expected image_labels to be 1D, got {image_labels.shape}")
    if pred_maps.ndim != 3:
        raise ValueError(f"Expected pred_maps to be 3D [N,H,W], got {pred_maps.shape}")
    if gt_masks.ndim != 3:
        raise ValueError(f"Expected gt_masks to be 3D [N,H,W], got {gt_masks.shape}")

    n = image_scores.shape[0]
    if image_labels.shape[0] != n:
        raise ValueError("image_scores and image_labels length mismatch")
    if pred_maps.shape[0] != n:
        raise ValueError("image_scores and pred_maps batch size mismatch")
    if gt_masks.shape[0] != n:
        raise ValueError("image_scores and gt_masks batch size mismatch")


def _build_dataset_row(
    dataset_name: str,
    n_images: int,
    image_labels: np.ndarray,
    image_metrics,
    image_aupro: float,
    pixel_metrics,
    timing: dict[str, float],
) -> dict[str, Any]:
    return {
        "dataset": dataset_name,
        "num_images": int(n_images),
        "num_anomalous_images": int(image_labels.sum()),
        "image_auroc": image_metrics.auroc,
        "image_ap": image_metrics.ap,
        "image_f1": image_metrics.f1,
        "image_precision": image_metrics.precision,
        "image_recall": image_metrics.recall,
        "image_accuracy": image_metrics.accuracy,
        "image_weighted_accuracy": image_metrics.weighted_accuracy,
        "image_threshold": image_metrics.threshold,
        "image_tp": image_metrics.tp,
        "image_fp": image_metrics.fp,
        "image_fn": image_metrics.fn,
        "image_tn": image_metrics.tn,
        "image_aupro": image_aupro,
        "pixel_auroc": pixel_metrics.binary.auroc,
        "pixel_ap": pixel_metrics.binary.ap,
        "pixel_aupro": pixel_metrics.aupro,
        "pixel_f1": pixel_metrics.binary.f1,
        "pixel_precision": pixel_metrics.binary.precision,
        "pixel_recall": pixel_metrics.binary.recall,
        "pixel_accuracy": pixel_metrics.binary.accuracy,
        "pixel_weighted_accuracy": pixel_metrics.binary.weighted_accuracy,
        "pixel_threshold": pixel_metrics.binary.threshold,
        "pixel_tp": pixel_metrics.binary.tp,
        "pixel_fp": pixel_metrics.binary.fp,
        "pixel_fn": pixel_metrics.binary.fn,
        "pixel_tn": pixel_metrics.binary.tn,
        "time_fit_s": timing.get("fit", float("nan")),
        "time_calibration_s": timing.get("calibration", float("nan")),
        "time_infer_s": timing.get("infer", float("nan")),
        "time_total_s": timing.get("total", float("nan")),
    }


def _build_per_image_rows(
    dataset_name: str,
    image_scores: np.ndarray,
    image_labels: np.ndarray,
    gt_masks: np.ndarray,
    pred_maps: np.ndarray,
    image_threshold: float,
    pixel_threshold: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for index in range(image_scores.shape[0]):
        gt_mask = gt_masks[index] > 0
        pred_map = pred_maps[index]
        pred_binary = pred_map >= pixel_threshold

        rows.append(
            {
                "dataset": dataset_name,
                "image_index": int(index),
                "image_label": int(image_labels[index]),
                "image_score": float(image_scores[index]),
                "image_pred": int(image_scores[index] >= image_threshold),
                "pixel_gt_positive_ratio": float(gt_mask.mean()),
                "pixel_pred_positive_ratio": float(pred_binary.mean()),
                "pixel_map_max": float(pred_map.max()),
                "pixel_map_mean": float(pred_map.mean()),
            }
        )

    return rows


def _select_artifact_indices(
    save_mode: str,
    image_labels: np.ndarray,
    image_scores: np.ndarray,
    max_images: int | None,
) -> set[int]:
    total = image_scores.shape[0]

    if save_mode == "all":
        return set(range(total))

    if save_mode == "anomalous":
        anomalous_indices = np.flatnonzero(image_labels > 0).astype(np.int64, copy=False)
        return {int(idx) for idx in anomalous_indices}

    if save_mode == "top_k":
        if max_images is None:
            raise ValueError("max_images must be provided for top_k save mode")
        sorted_idx = np.argsort(image_scores)[::-1].astype(np.int64, copy=False)
        return {int(idx) for idx in sorted_idx[:max_images]}

    raise ValueError(f"Unsupported artifact save_mode: {save_mode}")


def _save_dataset_artifacts_impl(
    dataset_name: str,
    test_loader,
    image_labels: np.ndarray,
    pred_maps: np.ndarray,
    gt_masks: np.ndarray,
    selected_indices: set[int],
    pixel_threshold: float,
    overlay_alpha: float,
    artifacts_root: Path,
    *,
    denormalize_image_fn: Any,
    save_prediction_artifacts_fn: Any,
) -> None:
    if not selected_indices:
        LOGGER.info("No artifact samples selected for dataset=%s", dataset_name)
        return

    current_index = 0
    saved = 0
    total_to_save = len(selected_indices)

    with create_progress_bar(
        test_loader,
        desc=artifacts_progress_desc(dataset_name),
    ) as artifact_iterator:
        for batch_idx, batch in enumerate(artifact_iterator, start=1):
            images = _extract_images(batch)
            batch_size = int(images.shape[0])

            for local_idx in range(batch_size):
                global_idx = current_index + local_idx
                if global_idx not in selected_indices:
                    continue

                image_rgb_u8 = denormalize_image_fn(
                    images[local_idx].detach().cpu().numpy()
                )
                gt_mask = gt_masks[global_idx]
                pred_map = pred_maps[global_idx]

                label_name = "anomalous" if int(image_labels[global_idx]) == 1 else "good"
                sample_dir = (
                    artifacts_root
                    / dataset_name
                    / label_name
                    / f"sample_{global_idx:05d}"
                )

                save_prediction_artifacts_fn(
                    output_dir=sample_dir,
                    image_rgb_u8=image_rgb_u8,
                    gt_mask=gt_mask,
                    pred_score_map=pred_map,
                    pixel_threshold=pixel_threshold,
                    overlay_alpha=overlay_alpha,
                )
                saved += 1

            current_index += batch_size
            artifact_iterator.set_postfix(
                make_progress_postfix(
                    batch=batch_idx,
                    batch_size=batch_size,
                    saved=saved,
                    total=total_to_save,
                    phase="artifacts",
                ),
                refresh=False,
            )


def _extract_images(batch) -> torch.Tensor:
    if isinstance(batch, dict):
        image_batch = batch.get("image")
        if image_batch is None:
            raise ValueError("Batch dictionary is missing 'image' key")
        if not isinstance(image_batch, torch.Tensor):
            raise ValueError("Batch['image'] must be a torch.Tensor")
        return image_batch

    if isinstance(batch, (tuple, list)) and batch:
        image_batch = batch[0]
        if not isinstance(image_batch, torch.Tensor):
            raise ValueError("Batch[0] must be a torch.Tensor")
        return image_batch

    if isinstance(batch, torch.Tensor):
        return batch

    raise ValueError("Unsupported batch format for artifact extraction")


def _append_mean_row(dataset_df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = dataset_df.select_dtypes(include=[np.number]).columns.tolist()
    mean_row: dict[str, Any] = {"dataset": "MEAN"}

    for col in dataset_df.columns:
        if col in numeric_cols:
            mean_row[col] = float(dataset_df[col].mean(skipna=True))
        elif col != "dataset":
            mean_row[col] = np.nan

    return pd.concat([dataset_df, pd.DataFrame([mean_row])], ignore_index=True)


@contextmanager
def _profile_phase(profiler: DatasetProfiler, phase_name: str) -> Iterator[None]:
    """Context manager to measure one named phase in the dataset profiler."""
    profiler.start_phase()
    try:
        yield
    finally:
        profiler.end_phase(phase_name)
