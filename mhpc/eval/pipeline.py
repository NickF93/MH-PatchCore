"""Execution pipeline for MH-PatchCore experiments."""

from __future__ import annotations

import gc
import logging
from collections.abc import Sized
from typing import Any, cast

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import torch

from mhpc.core.plugins.dataloader.contracts import DataLoaderPlugin
from mhpc.core.mh_patch_core import MHPatchCore
from mhpc.core.plugins.plugin_chain import build_runtime_plugin_chain
from mhpc.eval.artifacts import denormalize_image, save_prediction_artifacts
from mhpc.eval.config import RunConfig
from mhpc.eval.data_loading import (
    build_dataset_loaders,
    resolve_dataset_plan,
)
from mhpc.eval.frozen_replay import run_frozen_test_eval, run_frozen_train_replay
from mhpc.eval.metrics import (
    compute_binary_metrics,
    compute_image_aupro,
    compute_pixel_metrics,
)
from mhpc.eval.pipeline_helpers import (
    _append_mean_row,
    _build_calibration_train_loader as _build_calibration_train_loader_impl,
    _build_dataset_row,
    _build_model_impl,
    _build_per_image_rows,
    _fit_score_calibrator,
    _make_run_timestamp,
    _profile_phase,
    _resolve_device,
    _save_dataset_artifacts_impl,
    _select_artifact_indices,
    _validate_prediction_shapes,
    _validate_strict_streaming_pipeline_contract,
)
from mhpc.eval.profiling import (
    PHASE_RECORD_COLUMNS,
    RESOURCE_SAMPLE_COLUMNS,
    DatasetProfiler,
    phases_to_timing_dict,
)
from mhpc.eval.reproducibility import (
    apply_reproducibility_preamble,
    build_reproducibility_preamble,
)
from mhpc.eval.teacher_export import (
    save_teacher_checkpoint,
    save_teacher_memory_bank_artifact,
    teacher_checkpoint_enabled,
    teacher_memory_bank_enabled,
    teacher_replay_enabled,
)
from mhpc.util.param_binding import build_plugin_bind_context
from mhpc.util.progress import (
    DATASETS_PROGRESS_DESC,
    configure_progress_rendering,
    create_progress_bar,
    make_progress_postfix,
    metrics_progress_desc,
)

LOGGER = logging.getLogger(__name__)


def _print_dataset_metrics_row(row: dict[str, Any]) -> None:
    table = pd.DataFrame([row], columns=list(row)).to_string(index=False)
    print(
        f"\nDataset metrics [{row['dataset']}]:\n{table}",
        flush=True,
    )


def _build_model(
    config: RunConfig,
    device: torch.device,
    plugin_bundle=None,
):
    """Build the runtime model through the pipeline-owned construction hook."""
    return _build_model_impl(
        config=config,
        device=device,
        model_cls=MHPatchCore,
        plugin_bundle=plugin_bundle,
    )


def _build_calibration_train_loader(
    config: RunConfig,
    dataset_name: str,
    dataloader_plugin: DataLoaderPlugin,
):
    """Build the calibration loader through the pipeline-owned helper hook."""
    return _build_calibration_train_loader_impl(
        config=config,
        dataset_name=dataset_name,
        dataloader_plugin=dataloader_plugin,
    )


def _save_dataset_artifacts(
    dataset_name: str,
    test_loader,
    image_labels: np.ndarray,
    pred_maps: np.ndarray,
    gt_masks: np.ndarray,
    selected_indices: set[int],
    pixel_threshold: float,
    overlay_alpha: float,
    artifacts_root,
) -> None:
    """Save rendered sample artifacts through the pipeline-owned writer hook."""
    _save_dataset_artifacts_impl(
        dataset_name=dataset_name,
        test_loader=test_loader,
        image_labels=image_labels,
        pred_maps=pred_maps,
        gt_masks=gt_masks,
        selected_indices=selected_indices,
        pixel_threshold=pixel_threshold,
        overlay_alpha=overlay_alpha,
        artifacts_root=artifacts_root,
        denormalize_image_fn=denormalize_image,
        save_prediction_artifacts_fn=save_prediction_artifacts,
    )


def run_experiment(config: RunConfig) -> pd.DataFrame:
    """Run all configured datasets and persist metrics/artifacts.

    Args:
        config: Validated run configuration.

    Returns:
        Summary dataframe including per-dataset rows and a final mean row.

    Raises:
        RuntimeError: If no dataset could be evaluated.
    """
    configure_progress_rendering(
        enabled=config.render.progress.enabled,
        leave=config.render.progress.leave,
        dynamic_ncols=config.render.progress.dynamic_ncols,
        min_interval=config.render.progress.min_interval,
    )
    if config.runtime.device == "cpu":
        device = torch.device("cpu")
    else:
        device = _resolve_device()
    LOGGER.info("Resolved execution device: %s", device)
    _validate_strict_streaming_pipeline_contract(config)

    reproducibility_preamble = build_reproducibility_preamble(
        seed=config.experiment.seed,
        device=device,
    )
    apply_reproducibility_preamble(reproducibility_preamble)
    bind_context = build_plugin_bind_context(
        training_contract=config.training.contract,
        seed=reproducibility_preamble.seed,
    )
    runtime_plugin_chain = build_runtime_plugin_chain(
        selection_map=config.plugins.as_selection_map(),
        bind_context=bind_context,
        slot_params_map=config.slot_params,
    )
    dataloader_plugin = runtime_plugin_chain.dataloader_plugin

    base_output = (
        config.paths.output_root / config.experiment.name / _make_run_timestamp()
    )
    metrics_dir = base_output / "metrics"
    artifacts_root = base_output / "artifacts"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    artifacts_root.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Run output directory: %s", base_output)

    profiler = DatasetProfiler(device=device)
    dataset_rows: list[dict[str, Any]] = []
    per_image_rows: list[dict[str, Any]] = []
    profiling_sample_rows: list[dict[str, Any]] = []
    profiling_timing_rows: list[dict[str, Any]] = []

    dataset_plan = resolve_dataset_plan(dataloader_plugin)
    with create_progress_bar(
        dataset_plan,
        desc=DATASETS_PROGRESS_DESC,
    ) as dataset_iterator:
        datasets_total = len(dataset_plan)
        for dataset_idx, dataset_name in enumerate(dataset_iterator, start=1):
            LOGGER.info("Processing dataset=%s", dataset_name)
            dataset_iterator.set_postfix(
                make_progress_postfix(
                    batch=dataset_idx,
                    total=datasets_total,
                    phase=f"{dataset_name}:load",
                ),
                refresh=False,
            )
            # Align with patchcore-inspection reproducibility semantics:
            # each dataset starts from the same RNG state.
            apply_reproducibility_preamble(reproducibility_preamble)

            train_loader, test_loader = build_dataset_loaders(
                dataset_name=dataset_name,
                dataset_idx=dataset_idx,
                dataloader_plugin=dataloader_plugin,
            )

            if len(cast(Sized, test_loader.dataset)) == 0:
                LOGGER.warning("Empty test split; skipping dataset=%s", dataset_name)
                dataset_iterator.set_postfix(
                    make_progress_postfix(
                        batch=dataset_idx,
                        total=datasets_total,
                        phase=f"{dataset_name}:skipped_empty_test",
                    ),
                    refresh=False,
                )
                continue

            model = _build_model(
                config=config,
                device=device,
                plugin_bundle=runtime_plugin_chain,
            )
            profiler.start_dataset(dataset_name)
            profiling_finished = False
            try:
                LOGGER.info("Fitting model for dataset=%s", dataset_name)
                dataset_iterator.set_postfix(
                    make_progress_postfix(
                        batch=dataset_idx,
                        total=datasets_total,
                        phase=f"{dataset_name}:fit",
                    ),
                    refresh=False,
                )
                with _profile_phase(profiler, "fit"):
                    model.fit(train_loader)

                if teacher_checkpoint_enabled(config):
                    dataset_iterator.set_postfix(
                        make_progress_postfix(
                            batch=dataset_idx,
                            total=datasets_total,
                            phase=f"{dataset_name}:teacher_checkpoint",
                        ),
                        refresh=False,
                    )
                    with _profile_phase(profiler, "teacher_checkpoint"):
                        save_teacher_checkpoint(
                            config=config,
                            model=model,
                            dataset_name=dataset_name,
                            train_loader=train_loader,
                            artifacts_root=artifacts_root,
                        )

                if teacher_memory_bank_enabled(config):
                    dataset_iterator.set_postfix(
                        make_progress_postfix(
                            batch=dataset_idx,
                            total=datasets_total,
                            phase=f"{dataset_name}:memory_bank_artifact",
                        ),
                        refresh=False,
                    )
                    with _profile_phase(profiler, "memory_bank_artifact"):
                        save_teacher_memory_bank_artifact(
                            config=config,
                            model=model,
                            dataset_name=dataset_name,
                            train_loader=train_loader,
                            artifacts_root=artifacts_root,
                        )

                if teacher_replay_enabled(config):
                    dataset_iterator.set_postfix(
                        make_progress_postfix(
                            batch=dataset_idx,
                            total=datasets_total,
                            phase=f"{dataset_name}:teacher_replay",
                        ),
                        refresh=False,
                    )
                    with _profile_phase(profiler, "teacher_replay"):
                        run_frozen_train_replay(
                            config=config,
                            model=model,
                            dataset_name=dataset_name,
                            train_loader=train_loader,
                            artifacts_root=artifacts_root,
                        )

                dataset_iterator.set_postfix(
                    make_progress_postfix(
                        batch=dataset_idx,
                        total=datasets_total,
                        phase=f"{dataset_name}:calibration",
                    ),
                    refresh=False,
                )
                calibration_loader = train_loader
                if config.evaluation.calibration.mode != "none":
                    calibration_loader = _build_calibration_train_loader(
                        config=config,
                        dataset_name=dataset_name,
                        dataloader_plugin=dataloader_plugin,
                    )
                with _profile_phase(profiler, "calibration"):
                    score_calibrator = _fit_score_calibrator(
                        config=config,
                        model=model,
                        train_loader=calibration_loader,
                        dataset_name=dataset_name,
                    )

                LOGGER.info("Running inference for dataset=%s", dataset_name)
                dataset_iterator.set_postfix(
                    make_progress_postfix(
                        batch=dataset_idx,
                        total=datasets_total,
                        phase=f"{dataset_name}:infer",
                    ),
                    refresh=False,
                )
                with _profile_phase(profiler, "infer"):
                    prediction = model.infer_dataloader(test_loader)

                if teacher_replay_enabled(config):
                    dataset_iterator.set_postfix(
                        make_progress_postfix(
                            batch=dataset_idx,
                            total=datasets_total,
                            phase=f"{dataset_name}:teacher_eval",
                        ),
                        refresh=False,
                    )
                    with _profile_phase(profiler, "teacher_eval"):
                        run_frozen_test_eval(
                            config=config,
                            model=model,
                            dataset_name=dataset_name,
                            test_loader=test_loader,
                            artifacts_root=artifacts_root,
                        )

                image_scores_arr = np.asarray(prediction.image_scores, dtype=np.float64)
                image_labels_arr = np.asarray(prediction.image_labels, dtype=np.int32)
                pred_maps_arr = np.asarray(prediction.pred_maps, dtype=np.float64)
                gt_masks_arr = np.asarray(prediction.gt_masks, dtype=np.float64)

                _validate_prediction_shapes(
                    image_scores=image_scores_arr,
                    image_labels=image_labels_arr,
                    pred_maps=pred_maps_arr,
                    gt_masks=gt_masks_arr,
                )
                if score_calibrator is not None:
                    image_scores_arr, pred_maps_arr = score_calibrator.transform(
                        image_scores=image_scores_arr,
                        pixel_maps=pred_maps_arr,
                    )
                    _validate_prediction_shapes(
                        image_scores=image_scores_arr,
                        image_labels=image_labels_arr,
                        pred_maps=pred_maps_arr,
                        gt_masks=gt_masks_arr,
                    )

                aupro_cfg = config.evaluation.pixel_metrics.aupro
                metric_steps = 2 + int(aupro_cfg.image_enabled)
                with _profile_phase(profiler, "metrics"):
                    with create_progress_bar(
                        total=metric_steps,
                        desc=metrics_progress_desc(dataset_name),
                    ) as metrics_iterator:
                        image_metrics = compute_binary_metrics(
                            labels=image_labels_arr,
                            scores=image_scores_arr,
                            threshold_policy=config.evaluation.threshold_policy.image,
                        )
                        metrics_iterator.update(1)
                        metrics_iterator.set_postfix(
                            make_progress_postfix(
                                batch=1,
                                total=metric_steps,
                                phase="image_metrics",
                            ),
                            refresh=False,
                        )

                        if aupro_cfg.image_enabled:
                            image_aupro = compute_image_aupro(
                                labels=image_labels_arr,
                                scores=image_scores_arr,
                                max_fpr=aupro_cfg.max_fpr,
                                num_thresholds=aupro_cfg.num_thresholds,
                            )
                            metrics_iterator.update(1)
                            metrics_iterator.set_postfix(
                                make_progress_postfix(
                                    batch=2,
                                    total=metric_steps,
                                    phase="image_aupro",
                                ),
                                refresh=False,
                            )
                        else:
                            image_aupro = float("nan")

                        pixel_metrics = compute_pixel_metrics(
                            gt_masks=gt_masks_arr,
                            pred_maps=pred_maps_arr,
                            threshold_policy=config.evaluation.threshold_policy.pixel,
                            aupro_max_fpr=aupro_cfg.max_fpr,
                            aupro_num_thresholds=aupro_cfg.num_thresholds,
                            compute_aupro_enabled=aupro_cfg.pixel_enabled,
                        )
                        metrics_iterator.update(1)
                        metrics_iterator.set_postfix(
                            make_progress_postfix(
                                batch=metric_steps,
                                total=metric_steps,
                                phase="pixel_metrics",
                            ),
                            refresh=False,
                        )

                per_image_rows.extend(
                    _build_per_image_rows(
                        dataset_name=dataset_name,
                        image_scores=image_scores_arr,
                        image_labels=image_labels_arr,
                        gt_masks=gt_masks_arr,
                        pred_maps=pred_maps_arr,
                        image_threshold=image_metrics.threshold,
                        pixel_threshold=pixel_metrics.binary.threshold,
                    )
                )

                if config.artifacts.enabled:
                    selected_indices = _select_artifact_indices(
                        save_mode=config.artifacts.save_mode,
                        image_labels=image_labels_arr,
                        image_scores=image_scores_arr,
                        max_images=config.artifacts.max_images_per_dataset,
                    )
                    dataset_iterator.set_postfix(
                        make_progress_postfix(
                            batch=dataset_idx,
                            total=datasets_total,
                            phase=(
                                f"{dataset_name}:artifacts_to_save="
                                f"{len(selected_indices)}"
                            ),
                        ),
                        refresh=False,
                    )
                    with _profile_phase(profiler, "artifacts"):
                        _save_dataset_artifacts(
                            dataset_name=dataset_name,
                            test_loader=test_loader,
                            image_labels=image_labels_arr,
                            pred_maps=pred_maps_arr,
                            gt_masks=gt_masks_arr,
                            selected_indices=selected_indices,
                            pixel_threshold=pixel_metrics.binary.threshold,
                            overlay_alpha=config.artifacts.overlay_alpha,
                            artifacts_root=artifacts_root,
                        )

                samples, phases = profiler.finish_dataset()
                profiling_finished = True
                profiling_sample_rows.extend(vars(sample) for sample in samples)
                profiling_timing_rows.extend(vars(phase) for phase in phases)
                timing = phases_to_timing_dict(phases)

                dataset_row = _build_dataset_row(
                    dataset_name=dataset_name,
                    n_images=image_scores_arr.shape[0],
                    image_labels=image_labels_arr,
                    image_metrics=image_metrics,
                    image_aupro=image_aupro,
                    pixel_metrics=pixel_metrics,
                    timing=timing,
                )
                dataset_rows.append(dataset_row)
                _print_dataset_metrics_row(dataset_row)

                dataset_iterator.set_postfix(
                    make_progress_postfix(
                        batch=dataset_idx,
                        total=datasets_total,
                        images=int(image_scores_arr.shape[0]),
                        phase=f"{dataset_name}:done",
                    ),
                    refresh=False,
                )
            finally:
                if not profiling_finished:
                    try:
                        profiler.finish_dataset()
                    except Exception as exc:
                        LOGGER.debug(
                            "Profiler finalization skipped after dataset failure: "
                            "dataset=%s error=%s",
                            dataset_name,
                            exc,
                        )

                del model
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    if not dataset_rows:
        raise RuntimeError("No datasets were evaluated. Check dataset paths/configuration.")

    dataset_df = pd.DataFrame(dataset_rows)
    summary_df = _append_mean_row(dataset_df)
    per_image_df = pd.DataFrame(per_image_rows)

    dataset_csv = metrics_dir / "per_dataset.csv"
    summary_csv = metrics_dir / "summary.csv"
    per_image_csv = metrics_dir / "per_image.csv"
    profiling_samples_csv = metrics_dir / "profiling_samples.csv"
    profiling_timings_csv = metrics_dir / "profiling_timings.csv"

    dataset_df.to_csv(dataset_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    per_image_df.to_csv(per_image_csv, index=False)
    pd.DataFrame(
        profiling_sample_rows,
        columns=list(RESOURCE_SAMPLE_COLUMNS),
    ).to_csv(profiling_samples_csv, index=False)
    pd.DataFrame(
        profiling_timing_rows,
        columns=list(PHASE_RECORD_COLUMNS),
    ).to_csv(profiling_timings_csv, index=False)

    LOGGER.info("Saved dataset metrics to %s", dataset_csv)
    LOGGER.info("Saved summary metrics to %s", summary_csv)
    LOGGER.info("Saved per-image metrics to %s", per_image_csv)
    LOGGER.info("Saved profiling samples to %s", profiling_samples_csv)
    LOGGER.info("Saved profiling timings to %s", profiling_timings_csv)

    return summary_df
