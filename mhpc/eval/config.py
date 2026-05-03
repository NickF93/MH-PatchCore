"""Configuration schema and loader for MH-PatchCore experiments.

This module defines a strict hard-breaking schema with canonical
`pipeline.slots` + `pipeline.training` orchestration surfaces.
Validation is fail-fast and explicit to keep experiment execution
deterministic and auditable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml  # type: ignore[import-untyped]

from mhpc.eval.config_primitives import (
    _require_bool,
    _require_float,
    _require_int,
    _require_mapping,
    _require_non_empty_string,
    _require_optional_bool,
    _require_optional_float,
    _require_optional_mapping,
    _require_optional_string,
)
from mhpc.eval.slot_param_ownership import (
    canonical_slots,
)
from mhpc.util.repo_paths import resolve_repo_root

_FORBIDDEN_PLUGIN_ID = "default"
_PROJECT_ROOT = resolve_repo_root(__file__)
_CANONICAL_PLUGIN_STAGES: tuple[str, ...] = canonical_slots()
_EXPLICIT_TRAINABLE_STAGES: tuple[str, ...] = (
    "feature_agg",
    "proj1",
    "transform",
    "proj2",
    "mem_agg",
)
_TRAINING_CONTRACT_VALUES: frozenset[str] = frozenset({"OFFLINE", "STREAMING"})


@dataclass(frozen=True)
class ExperimentConfig:
    """Top-level experiment identity and reproducibility settings."""

    name: str
    seed: int


@dataclass(frozen=True)
class PathsConfig:
    """Filesystem paths used by the experiment."""

    output_root: Path


@dataclass(frozen=True)
class RuntimeConfig:
    """Runtime execution policy."""

    device: str


@dataclass(frozen=True)
class ThresholdPolicyConfig:
    """Threshold selection strategy for image/pixel predictions."""

    image: str
    pixel: str


@dataclass(frozen=True)
class AUPROConfig:
    """AUPRO numerical integration settings."""

    max_fpr: float
    num_thresholds: int
    image_enabled: bool
    pixel_enabled: bool


@dataclass(frozen=True)
class PixelMetricConfig:
    """Pixel-level metrics configuration."""

    aupro: AUPROConfig


@dataclass(frozen=True)
class ScoreCalibrationConfig:
    """Score calibration settings for evaluation outputs."""

    mode: str
    eps: float
    apply_to_image: bool
    apply_to_pixel: bool


@dataclass(frozen=True)
class EvaluationConfig:
    """Evaluation and reporting policy."""

    threshold_policy: ThresholdPolicyConfig
    pixel_metrics: PixelMetricConfig
    calibration: ScoreCalibrationConfig


@dataclass(frozen=True)
class ArtifactConfig:
    """Artifact generation and rendering settings."""

    enabled: bool
    split: str
    save_mode: str
    max_images_per_dataset: int | None
    overlay_alpha: float


@dataclass(frozen=True)
class ProgressRenderConfig:
    """Progress-bar rendering policy."""

    enabled: bool
    leave: bool
    dynamic_ncols: bool
    min_interval: float


@dataclass(frozen=True)
class LoggingRenderConfig:
    """Logging rendering policy."""

    level: str


@dataclass(frozen=True)
class RenderConfig:
    """Render policy for progress bars and logger output."""

    progress: ProgressRenderConfig
    logging: LoggingRenderConfig


@dataclass(frozen=True)
class PluginSelectionConfig:
    """Config-driven plugin selection for currently pluginized slots."""

    dataloader: str
    backbone: str
    patch_align: str
    preprocess: str
    feature_agg: str
    proj1: str
    transform: str
    proj2: str
    mem_agg: str
    materialize: str
    distance: str
    scoring: str

    def as_selection_map(self) -> dict[str, str]:
        """Return plugin selection as a stage->plugin mapping."""
        return {
            "dataloader": self.dataloader,
            "backbone": self.backbone,
            "patch_align": self.patch_align,
            "preprocess": self.preprocess,
            "feature_agg": self.feature_agg,
            "proj1": self.proj1,
            "transform": self.transform,
            "proj2": self.proj2,
            "mem_agg": self.mem_agg,
            "materialize": self.materialize,
            "distance": self.distance,
            "scoring": self.scoring,
        }


@dataclass(frozen=True)
class PipelineTrainingConfig:
    """TRAIN/INFERENCE orchestration metadata parsed from pipeline.training."""

    contract: str
    fit_epochs: dict[str, int]


@dataclass(frozen=True)
class RunConfig:
    """Complete validated run configuration."""

    experiment: ExperimentConfig
    paths: PathsConfig
    runtime: RuntimeConfig
    training: PipelineTrainingConfig
    evaluation: EvaluationConfig
    artifacts: ArtifactConfig
    render: RenderConfig
    slot_params: dict[str, dict[str, Any]]
    plugins: PluginSelectionConfig


def load_run_config(config_path: str | Path) -> RunConfig:
    """Load and validate a run configuration.

    Args:
        config_path: YAML configuration file path.

    Returns:
        A validated :class:`RunConfig` instance.

    Raises:
        FileNotFoundError: If the file is missing.
        ValueError: If required keys or values are invalid.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        raw_obj = yaml.safe_load(handle)

    if not isinstance(raw_obj, Mapping):
        raise ValueError("Configuration root must be a YAML mapping")

    if "schema_version" in raw_obj:
        raise ValueError(
            "schema_version is not supported in canonical config surface. "
            "Remove schema_version and use non-versioned "
            "pipeline.training + pipeline.slots layout."
        )

    if "pipeline" not in raw_obj:
        raise ValueError(
            "pipeline.slots block is required and must include all canonical stages: "
            "dataloader, backbone, patch_align, preprocess, feature_agg, proj1, "
            "transform, proj2, mem_agg, materialize, distance, scoring"
        )
    pipeline_cfg = _require_mapping(raw_obj, "pipeline")
    training_cfg = _parse_pipeline_training(pipeline_cfg)
    slot_params_by_stage = _parse_pipeline_slot_params(pipeline_cfg)
    _reject_legacy_slot_owned_parameter_surfaces(raw_obj)

    experiment_cfg = _parse_experiment(_require_mapping(raw_obj, "experiment"))
    paths_cfg = _parse_paths(_require_mapping(raw_obj, "paths"))
    runtime_cfg = _parse_runtime(_require_mapping(raw_obj, "runtime"))
    evaluation_cfg = _parse_evaluation(_require_mapping(raw_obj, "evaluation"))
    artifacts_cfg = _parse_artifacts(_require_mapping(raw_obj, "artifacts"))
    render_cfg = _parse_render(_require_optional_mapping(raw_obj, "render"))
    plugins_cfg = _parse_plugin_selection(
        _require_optional_mapping(raw_obj, "plugins"),
        pipeline_cfg=pipeline_cfg,
    )

    if training_cfg.contract == "STREAMING":
        if evaluation_cfg.calibration.mode not in {"none", "zscore"}:
            raise ValueError(
                "evaluation.calibration.mode must be 'none' or 'zscore' when "
                "pipeline.training.contract is STREAMING "
                "(strict streaming contract)."
            )

    return RunConfig(
        experiment=experiment_cfg,
        paths=paths_cfg,
        runtime=runtime_cfg,
        training=training_cfg,
        evaluation=evaluation_cfg,
        artifacts=artifacts_cfg,
        render=render_cfg,
        slot_params={
            stage_name: dict(params)
            for stage_name, params in slot_params_by_stage.items()
        },
        plugins=plugins_cfg,
    )


def _reject_legacy_slot_owned_parameter_surfaces(cfg: Mapping[str, Any]) -> None:
    """Fail fast when legacy slot-owned keys appear outside slot params."""
    forbidden_surface_paths = _collect_legacy_slot_owned_surface_paths(cfg)
    if not forbidden_surface_paths:
        return

    forbidden_surface_path = forbidden_surface_paths[0]
    raise ValueError(
        "Legacy slot-owned parameter key is not allowed outside "
        f"pipeline.slots.<slot>.params: {forbidden_surface_path}"
    )


def _collect_legacy_slot_owned_surface_paths(
    cfg: Mapping[str, Any],
) -> tuple[str, ...]:
    paths: list[str] = []

    if "schema_version" in cfg:
        paths.append("schema_version")

    if "data" in cfg:
        data_cfg = _require_optional_mapping(cfg, "data")
        if data_cfg is None:
            paths.append("data")
        else:
            data_keys = sorted(str(key) for key in data_cfg.keys())
            if data_keys:
                paths.extend(f"data.{key}" for key in data_keys)
            else:
                paths.append("data")

    model_cfg = _require_optional_mapping(cfg, "model")
    if model_cfg is not None:
        patch_core_cfg = _require_optional_mapping(model_cfg, "patch_core")
        if patch_core_cfg is None:
            paths.append("model")
        else:
            if not patch_core_cfg:
                paths.append("model.patch_core")
            else:
                paths.extend(
                    f"model.patch_core.{key}" for key in sorted(patch_core_cfg.keys())
                )
        for key in sorted(set(model_cfg.keys()) - {"patch_core"}):
            paths.append(f"model.{key}")

    return tuple(sorted(paths))


def _parse_experiment(cfg: Mapping[str, Any]) -> ExperimentConfig:
    name = _require_non_empty_string(cfg, "name")
    seed = _require_int(cfg, "seed")
    return ExperimentConfig(name=name, seed=seed)


def _parse_paths(cfg: Mapping[str, Any]) -> PathsConfig:
    _ensure_allowed_keys(
        cfg,
        allowed_keys={"output_root"},
        context="paths",
    )
    output_root = _resolve_repo_relative_path(
        _require_non_empty_string(cfg, "output_root")
    )
    return PathsConfig(output_root=output_root)


def _parse_runtime(cfg: Mapping[str, Any]) -> RuntimeConfig:
    _ensure_allowed_keys(
        cfg,
        allowed_keys={"device"},
        context="runtime",
    )
    device = _require_non_empty_string(cfg, "device").strip().lower()
    if device not in {"auto", "cpu"}:
        raise ValueError("runtime.device must be one of: auto, cpu")
    return RuntimeConfig(device=device)


def _resolve_repo_relative_path(raw_value: str) -> Path:
    authored_path = Path(raw_value).expanduser()
    if authored_path.is_absolute():
        return authored_path
    return _PROJECT_ROOT / authored_path


def _parse_pipeline_training(
    pipeline_cfg: Mapping[str, Any],
) -> PipelineTrainingConfig:
    """Parse canonical pipeline.training orchestration metadata."""

    _ensure_allowed_keys(
        pipeline_cfg,
        allowed_keys={"training", "slots"},
        context="pipeline",
    )
    training_cfg = _require_mapping(pipeline_cfg, "training")
    _ensure_allowed_keys(
        training_cfg,
        allowed_keys={"contract", "fit_epochs"},
        context="pipeline.training",
    )

    contract_raw = _require_non_empty_string(training_cfg, "contract")
    contract = contract_raw.strip().upper()
    if contract not in _TRAINING_CONTRACT_VALUES:
        raise ValueError(
            "pipeline.training.contract must be one of: OFFLINE, STREAMING"
        )

    fit_epochs_cfg = _require_mapping(training_cfg, "fit_epochs")
    expected_stage_keys = set(_EXPLICIT_TRAINABLE_STAGES)
    authored_stage_keys = set(str(stage_name) for stage_name in fit_epochs_cfg.keys())
    missing_stage_keys = sorted(expected_stage_keys - authored_stage_keys)
    if missing_stage_keys:
        raise ValueError(
            "pipeline.training.fit_epochs must define exactly the explicit trainable "
            "stages feature_agg, proj1, transform, proj2, mem_agg; missing: "
            f"{', '.join(missing_stage_keys)}"
        )
    unsupported_stage_keys = sorted(authored_stage_keys - expected_stage_keys)
    if unsupported_stage_keys:
        raise ValueError(
            "pipeline.training.fit_epochs may define only the explicit trainable "
            "stages feature_agg, proj1, transform, proj2, mem_agg; unsupported: "
            f"{', '.join(unsupported_stage_keys)}"
        )

    stage_fit_epochs: dict[str, int] = {}
    for stage_name in _EXPLICIT_TRAINABLE_STAGES:
        value = _require_int(
            fit_epochs_cfg,
            stage_name,
        )
        if value <= 0:
            raise ValueError(
                "pipeline.training.fit_epochs values must be positive integers: "
                f"stage='{stage_name}' value={value}"
            )
        stage_fit_epochs[stage_name] = value

    return PipelineTrainingConfig(
        contract=contract,
        fit_epochs=stage_fit_epochs,
    )


def _parse_pipeline_slot_params(
    pipeline_cfg: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    slots_cfg = _require_mapping(pipeline_cfg, "slots")
    known_stages = set(_CANONICAL_PLUGIN_STAGES)
    unknown_stages = sorted(set(slots_cfg.keys()) - known_stages)
    if unknown_stages:
        raise ValueError(
            "pipeline.slots contains unsupported stages: "
            f"{', '.join(str(stage) for stage in unknown_stages)}"
        )

    missing_stages = [stage for stage in sorted(known_stages) if stage not in slots_cfg]
    if missing_stages:
        raise ValueError(
            "pipeline.slots is missing canonical stages: "
            f"{', '.join(missing_stages)}"
        )

    params_by_stage: dict[str, Mapping[str, Any]] = {}
    for stage_name in _CANONICAL_PLUGIN_STAGES:
        slot_cfg = _require_mapping(slots_cfg, stage_name)
        _ensure_allowed_keys(
            slot_cfg,
            allowed_keys={"plugin", "params"},
            context=f"pipeline.slots.{stage_name}",
        )
        params_cfg = _require_optional_mapping(slot_cfg, "params")
        if params_cfg is None:
            raise ValueError(
                f"pipeline.slots.{stage_name}.params is required and must be a mapping"
            )
        if stage_name in _EXPLICIT_TRAINABLE_STAGES and "fit_epochs" in params_cfg:
            raise ValueError(
                "Legacy slot fit_epochs is forbidden in pipeline.slots.<stage>.params; "
                f"use pipeline.training.fit_epochs.{stage_name}."
            )
        params_by_stage[stage_name] = params_cfg
    return params_by_stage


def _ensure_allowed_keys(
    mapping: Mapping[str, Any],
    *,
    allowed_keys: set[str],
    context: str,
) -> None:
    unknown_keys = sorted(str(key) for key in set(mapping.keys()) - allowed_keys)
    if unknown_keys:
        raise ValueError(
            f"{context} contains unsupported keys: {', '.join(unknown_keys)}"
        )



def _parse_evaluation(cfg: Mapping[str, Any]) -> EvaluationConfig:
    threshold_policy_cfg = _require_mapping(cfg, "threshold_policy")
    image_policy = _require_non_empty_string(threshold_policy_cfg, "image")
    pixel_policy = _require_non_empty_string(threshold_policy_cfg, "pixel")

    allowed_policies = {"best_f1_per_dataset", "fixed_0_5"}
    if image_policy not in allowed_policies:
        raise ValueError(
            "evaluation.threshold_policy.image must be one of: "
            f"{', '.join(sorted(allowed_policies))}"
        )
    if pixel_policy not in allowed_policies:
        raise ValueError(
            "evaluation.threshold_policy.pixel must be one of: "
            f"{', '.join(sorted(allowed_policies))}"
        )

    pixel_metrics_cfg = _require_mapping(cfg, "pixel_metrics")
    aupro_cfg = _require_mapping(pixel_metrics_cfg, "aupro")
    max_fpr = _require_float(aupro_cfg, "max_fpr")
    num_thresholds = _require_int(aupro_cfg, "num_thresholds")
    image_enabled = _require_optional_bool(aupro_cfg, "image_enabled", default=True)
    pixel_enabled = _require_optional_bool(aupro_cfg, "pixel_enabled", default=False)

    if not 0.0 < max_fpr <= 1.0:
        raise ValueError("evaluation.pixel_metrics.aupro.max_fpr must be in (0, 1]")
    if num_thresholds < 8:
        raise ValueError("evaluation.pixel_metrics.aupro.num_thresholds must be >= 8")
    calibration = _parse_calibration(_require_optional_mapping(cfg, "calibration"))

    return EvaluationConfig(
        threshold_policy=ThresholdPolicyConfig(image=image_policy, pixel=pixel_policy),
        pixel_metrics=PixelMetricConfig(
            aupro=AUPROConfig(
                max_fpr=max_fpr,
                num_thresholds=num_thresholds,
                image_enabled=image_enabled,
                pixel_enabled=pixel_enabled,
            )
        ),
        calibration=calibration,
    )


def _parse_calibration(cfg: Mapping[str, Any] | None) -> ScoreCalibrationConfig:
    raw_cfg: Mapping[str, Any]
    if cfg is None:
        raw_cfg = {}
    else:
        raw_cfg = cfg

    mode = _require_optional_string(raw_cfg, "mode", default="none").lower()
    allowed_modes = {"none", "zscore", "ecdf"}
    if mode not in allowed_modes:
        raise ValueError(
            "evaluation.calibration.mode must be one of: "
            f"{', '.join(sorted(allowed_modes))}"
        )

    eps = _require_optional_float(raw_cfg, "eps", default=1.0e-12)
    if eps <= 0.0:
        raise ValueError("evaluation.calibration.eps must be > 0")

    apply_to_image = _require_optional_bool(raw_cfg, "apply_to_image", default=True)
    apply_to_pixel = _require_optional_bool(raw_cfg, "apply_to_pixel", default=True)

    return ScoreCalibrationConfig(
        mode=mode,
        eps=eps,
        apply_to_image=apply_to_image,
        apply_to_pixel=apply_to_pixel,
    )


def _parse_artifacts(cfg: Mapping[str, Any]) -> ArtifactConfig:
    enabled = _require_bool(cfg, "enabled")
    split = _require_non_empty_string(cfg, "split")
    save_mode = _require_non_empty_string(cfg, "save_mode")
    overlay_alpha = _require_float(cfg, "overlay_alpha")

    max_images_raw = cfg.get("max_images_per_dataset")
    max_images_per_dataset: int | None
    if max_images_raw is None:
        max_images_per_dataset = None
    elif isinstance(max_images_raw, int):
        if max_images_raw <= 0:
            raise ValueError("artifacts.max_images_per_dataset must be > 0")
        max_images_per_dataset = max_images_raw
    else:
        raise ValueError("artifacts.max_images_per_dataset must be int or null")

    if split != "test":
        raise ValueError("artifacts.split currently supports only 'test'")

    allowed_modes = {"all", "anomalous", "top_k"}
    if save_mode not in allowed_modes:
        raise ValueError(
            "artifacts.save_mode must be one of: "
            f"{', '.join(sorted(allowed_modes))}"
        )
    if save_mode == "top_k" and max_images_per_dataset is None:
        raise ValueError(
            "artifacts.max_images_per_dataset is required when save_mode is 'top_k'"
        )

    if not 0.0 <= overlay_alpha <= 1.0:
        raise ValueError("artifacts.overlay_alpha must be in [0, 1]")

    return ArtifactConfig(
        enabled=enabled,
        split=split,
        save_mode=save_mode,
        max_images_per_dataset=max_images_per_dataset,
        overlay_alpha=overlay_alpha,
    )


def _parse_render(cfg: Mapping[str, Any] | None) -> RenderConfig:
    raw_cfg: Mapping[str, Any]
    if cfg is None:
        raw_cfg = {}
    else:
        raw_cfg = cfg

    progress_cfg = _parse_render_progress(_require_optional_mapping(raw_cfg, "progress"))
    logging_cfg = _parse_render_logging(_require_optional_mapping(raw_cfg, "logging"))
    return RenderConfig(progress=progress_cfg, logging=logging_cfg)

def _validate_concrete_public_plugin_ids(
    cfg: PluginSelectionConfig,
    *,
    surface_name: str,
) -> None:
    disallowed_stages = sorted(
        stage_name
        for stage_name, plugin_id in cfg.as_selection_map().items()
        if plugin_id.strip().lower() == _FORBIDDEN_PLUGIN_ID
    )
    if disallowed_stages:
        raise ValueError(
            f"{surface_name} must use concrete plugin ids; "
            "transitional plugin id 'default' is forbidden for stages: "
            f"{', '.join(disallowed_stages)}"
        )


def _parse_pipeline_slots_plugin_selection(
    *,
    slots_cfg: Mapping[str, Any],
    known_stages: set[str],
) -> PluginSelectionConfig:
    unknown_stages = sorted(set(slots_cfg.keys()) - known_stages)
    if unknown_stages:
        raise ValueError(
            "pipeline.slots contains unsupported stages: "
            f"{', '.join(str(stage) for stage in unknown_stages)}"
        )

    missing_stages = [stage for stage in sorted(known_stages) if stage not in slots_cfg]
    if missing_stages:
        raise ValueError(
            "pipeline.slots is missing canonical stages: "
            f"{', '.join(missing_stages)}"
        )

    slot_plugins: dict[str, str] = {}
    for stage_name in _CANONICAL_PLUGIN_STAGES:
        slot_cfg = _require_mapping(slots_cfg, stage_name)
        _ensure_allowed_keys(
            slot_cfg,
            allowed_keys={"plugin", "params"},
            context=f"pipeline.slots.{stage_name}",
        )
        slot_plugins[stage_name] = _require_non_empty_string(slot_cfg, "plugin")
        params_cfg = _require_optional_mapping(slot_cfg, "params")
        if params_cfg is None:
            raise ValueError(
                f"pipeline.slots.{stage_name}.params is required and must be a mapping"
            )

    parsed = PluginSelectionConfig(
        dataloader=slot_plugins["dataloader"],
        backbone=slot_plugins["backbone"],
        patch_align=slot_plugins["patch_align"],
        preprocess=slot_plugins["preprocess"],
        feature_agg=slot_plugins["feature_agg"],
        proj1=slot_plugins["proj1"],
        transform=slot_plugins["transform"],
        proj2=slot_plugins["proj2"],
        mem_agg=slot_plugins["mem_agg"],
        materialize=slot_plugins["materialize"],
        distance=slot_plugins["distance"],
        scoring=slot_plugins["scoring"],
    )
    _validate_concrete_public_plugin_ids(parsed, surface_name="pipeline.slots")
    return parsed


def _parse_plugin_selection(
    cfg: Mapping[str, Any] | None,
    *,
    pipeline_cfg: Mapping[str, Any] | None = None,
) -> PluginSelectionConfig:
    raw_cfg: Mapping[str, Any] | None = cfg
    known_stages = {
        "dataloader",
        "backbone",
        "patch_align",
        "preprocess",
        "feature_agg",
        "proj1",
        "transform",
        "proj2",
        "mem_agg",
        "materialize",
        "distance",
        "scoring",
    }

    slots_cfg: Mapping[str, Any] | None = None
    if pipeline_cfg is not None:
        slots_cfg = _require_optional_mapping(pipeline_cfg, "slots")
        if slots_cfg is None:
            raise ValueError(
                "pipeline.slots block is required when pipeline is defined and "
                "must include all canonical stages: "
                "dataloader, backbone, patch_align, preprocess, feature_agg, proj1, "
                "transform, proj2, mem_agg, materialize, distance, scoring"
            )

    if raw_cfg is not None:
        raise ValueError(
            "Legacy top-level plugins selection block is forbidden. "
            "Use pipeline.slots.<slot>.{plugin,params} for all canonical stages."
        )
    if slots_cfg is None:
        raise ValueError(
            "pipeline.slots block is required and must include all canonical stages: "
            "dataloader, backbone, patch_align, preprocess, feature_agg, proj1, "
            "transform, proj2, mem_agg, materialize, distance, scoring"
        )
    return _parse_pipeline_slots_plugin_selection(
        slots_cfg=slots_cfg,
        known_stages=known_stages,
    )


def _parse_render_progress(
    cfg: Mapping[str, Any] | None,
) -> ProgressRenderConfig:
    raw_cfg: Mapping[str, Any]
    if cfg is None:
        raw_cfg = {}
    else:
        raw_cfg = cfg

    enabled = _require_optional_bool(raw_cfg, "enabled", default=True)
    leave = _require_optional_bool(raw_cfg, "leave", default=True)
    dynamic_ncols = _require_optional_bool(raw_cfg, "dynamic_ncols", default=True)
    min_interval = _require_optional_float(raw_cfg, "min_interval", default=0.1)
    if min_interval <= 0.0:
        raise ValueError("render.progress.min_interval must be > 0")

    return ProgressRenderConfig(
        enabled=enabled,
        leave=leave,
        dynamic_ncols=dynamic_ncols,
        min_interval=min_interval,
    )


def _parse_render_logging(cfg: Mapping[str, Any] | None) -> LoggingRenderConfig:
    raw_cfg: Mapping[str, Any]
    if cfg is None:
        raw_cfg = {}
    else:
        raw_cfg = cfg

    level = _require_optional_string(raw_cfg, "level", default="INFO").upper()
    allowed_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
    if level not in allowed_levels:
        raise ValueError(
            "render.logging.level must be one of: "
            f"{', '.join(sorted(allowed_levels))}"
        )
    return LoggingRenderConfig(level=level)
