"""CLI-facing renderers and reproducibility helpers for experiment execution."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import subprocess
import textwrap

from mhpc.core.plugins.plugin_chain import build_runtime_plugin_chain
from mhpc.eval.config import RunConfig
from mhpc.util.param_binding import build_plugin_bind_context

_TRAIN_LABELS: tuple[tuple[str, str], ...] = (
    ("dataloader", "load"),
    ("backbone", "embed"),
    ("patch_align", "align"),
    ("preprocess", "pre"),
    ("feature_agg", "reduce.fit"),
    ("proj1", "proj1"),
    ("transform", "xfm.fit"),
    ("proj2", "proj2"),
    ("mem_agg", "mem.fit"),
    ("materialize", "bank"),
    ("distance", "dist.fit"),
    ("scoring", "score.fit"),
)
_INFER_LABELS: tuple[tuple[str, str], ...] = (
    ("dataloader", "load"),
    ("backbone", "embed"),
    ("patch_align", "align"),
    ("preprocess", "pre"),
    ("feature_agg", "reduce"),
    ("proj1", "proj1"),
    ("transform", "xfm"),
    ("proj2", "proj2"),
    ("distance", "dist.query"),
    ("scoring", "score"),
)
_BANNER_WIDTH = 108
_PARAM_SUMMARY_KEYS: dict[str, tuple[tuple[str, str], ...]] = {
    "dataloader": (
        ("batch", "batch_size"),
        ("img", "img_size"),
        ("augment", "train_augment_enabled"),
        ("policy", "streaming_augmentation_policy"),
    ),
    "backbone": (
        ("backbone", "backbone"),
        ("layers", "embedding_layers"),
    ),
    "patch_align": (
        ("patch", "patchsize"),
        ("stride", "patchstride"),
    ),
    "preprocess": (("pre_dim", "pretrain_embed_dimension"),),
    "feature_agg": (
        ("target_dim", "target_embed_dimension"),
        ("var", "pca_variance_ratio"),
    ),
    "transform": (
        ("precision", "storage_precision"),
        ("cov.method", "covariance_regularization.method"),
        ("shrink", "covariance_regularization.shrinkage"),
        ("floor", "covariance_regularization.eigen_floor_ratio"),
    ),
    "mem_agg": (
        ("gmm_k", "gmm_n_components"),
        ("clusters", "n_clusters"),
        ("mode", "kcenter_mode"),
        ("chunk", "kcenter_chunk_coreset_size"),
        ("coreset", "coreset_percentage"),
    ),
    "distance": (("k", "k"),),
    "scoring": (
        ("mode", "patch_scoring.mode"),
        ("paper_k", "paper_reweight_num_nn"),
        ("topk_k", "patch_scoring.pni.topk_k"),
        ("temp", "patch_scoring.pni.topk_temperature"),
        ("kernel", "patch_scoring.pni.neighborhood_kernel_size"),
    ),
}


@dataclass(frozen=True)
class _CompiledBannerStage:
    slot_name: str
    stage_label: str
    plugin_id: str


def render_loaded_pipeline_banner(
    *,
    cfg: RunConfig,
    config_path: Path,
) -> str:
    """Render a console banner for the exact loaded configuration only."""
    train_stages, infer_stages = _compile_loaded_pipeline_stages(cfg=cfg)
    lines = [
        "",
        _render_box_border("top", title="Loaded Config Compiled Pipeline"),
        *_render_box_lines(
            [
                f"Config: {config_path}",
                f"Experiment: {cfg.experiment.name}",
                f"Contract: {cfg.training.contract}",
            ]
        ),
        _render_box_border("section", title="TRAIN"),
        *_render_box_lines(
            _render_stage_section(
                stages=train_stages,
                slot_params_map=cfg.slot_params,
            )
        ),
        _render_box_border("section", title="INFER"),
        *_render_box_lines(
            _render_stage_section(
                stages=infer_stages,
                slot_params_map=cfg.slot_params,
            )
        ),
        _render_box_border("bottom"),
        "",
    ]
    return "\n".join(lines)


def save_run_reproducibility_artifacts(
    *,
    run_dir: Path,
    config_path: Path,
    repo_root: Path,
) -> None:
    """Persist reproducibility artifacts inside one concrete run directory."""
    (run_dir / "run_config.yaml").write_bytes(config_path.read_bytes())

    sha, dirty = _resolve_git_metadata(repo_root=repo_root)
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    (run_dir / "git_commit.txt").write_text(
        "\n".join(
            (
                f"git_commit_sha={sha}",
                f"git_dirty={dirty}",
                f"generated_at={generated_at}",
            )
        )
        + "\n",
        encoding="utf-8",
    )


def _compile_loaded_pipeline_stages(
    *,
    cfg: RunConfig,
) -> tuple[tuple[_CompiledBannerStage, ...], tuple[_CompiledBannerStage, ...]]:
    selection_map = cfg.plugins.as_selection_map()
    bind_context = build_plugin_bind_context(
        training_contract=cfg.training.contract,
        seed=cfg.experiment.seed,
    )
    runtime_bundle = build_runtime_plugin_chain(
        selection_map=selection_map,
        bind_context=bind_context,
        slot_params_map=cfg.slot_params,
    )
    selected_plugins = _selected_plugins_map(runtime_bundle)
    return (
        _compile_flow(
            labels=_TRAIN_LABELS,
            selection_map=selection_map,
            selected_plugins=selected_plugins,
            execution_mode="train",
        ),
        _compile_flow(
            labels=_INFER_LABELS,
            selection_map=selection_map,
            selected_plugins=selected_plugins,
            execution_mode="inference",
        ),
    )


def _selected_plugins_map(runtime_bundle: object) -> dict[str, object]:
    model_bundle = getattr(runtime_bundle, "model_plugin_bundle")
    return {
        "dataloader": getattr(runtime_bundle, "dataloader_plugin"),
        "backbone": getattr(model_bundle, "backbone_plugin"),
        "patch_align": getattr(model_bundle, "patch_align_plugin"),
        "preprocess": getattr(model_bundle, "preprocess_plugin"),
        "feature_agg": getattr(model_bundle, "feature_agg_plugin"),
        "proj1": getattr(model_bundle, "proj1_plugin"),
        "transform": getattr(model_bundle, "transform_plugin"),
        "proj2": getattr(model_bundle, "proj2_plugin"),
        "mem_agg": getattr(model_bundle, "mem_agg_plugin"),
        "materialize": getattr(model_bundle, "materialize_plugin"),
        "distance": getattr(model_bundle, "distance_plugin"),
        "scoring": getattr(model_bundle, "scoring_plugin"),
    }


def _compile_flow(
    *,
    labels: tuple[tuple[str, str], ...],
    selection_map: dict[str, str],
    selected_plugins: dict[str, object],
    execution_mode: str,
) -> tuple[_CompiledBannerStage, ...]:
    compiled: list[_CompiledBannerStage] = []
    for slot_name, stage_label in labels:
        plugin = selected_plugins[slot_name]
        if not _include_stage_in_mode(
            slot_name=slot_name,
            plugin=plugin,
            execution_mode=execution_mode,
        ):
            continue
        compiled.append(
            _CompiledBannerStage(
                slot_name=slot_name,
                stage_label=stage_label,
                plugin_id=selection_map[slot_name],
            )
        )
    return tuple(compiled)


def _include_stage_in_mode(
    *,
    slot_name: str,
    plugin: object,
    execution_mode: str,
) -> bool:
    if execution_mode == "train":
        return slot_name != "scoring" or bool(
            getattr(plugin, "requires_patch_scoring_state", False)
        )
    if execution_mode == "inference":
        return True
    raise ValueError(f"Unsupported execution_mode: {execution_mode!r}")


def _render_stage_section(
    *,
    stages: tuple[_CompiledBannerStage, ...],
    slot_params_map: dict[str, dict[str, object]],
) -> list[str]:
    lines: list[str] = []
    for idx, stage in enumerate(stages):
        lines.append(
            f"{stage.stage_label:<11} [{stage.slot_name}] │ {stage.plugin_id}"
        )
        summary = _summarize_slot_params(
            slot_name=stage.slot_name,
            params=slot_params_map.get(stage.slot_name, {}),
        )
        if summary is not None:
            lines.append(f"params      │ {summary}")
        if idx != len(stages) - 1:
            lines.append("            ▼")
    if not lines:
        return ["(no executable stages)"]
    return lines


def _summarize_slot_params(
    *,
    slot_name: str,
    params: dict[str, object],
) -> str | None:
    flattened = _flatten_params(params)
    if slot_name in {"proj1", "proj2", "materialize"} and not flattened:
        return None

    consumed: set[str] = set()
    parts: list[str] = []
    for label, dotted_key in _PARAM_SUMMARY_KEYS.get(slot_name, ()):
        value = flattened.get(dotted_key)
        if value is None:
            continue
        parts.append(f"{label}={value}")
        consumed.add(dotted_key)

    for dotted_key in sorted(flattened):
        if dotted_key in consumed:
            continue
        parts.append(f"{dotted_key}={flattened[dotted_key]}")

    if not parts:
        return None
    return " · ".join(parts)


def _flatten_params(
    params: dict[str, object],
    *,
    prefix: str = "",
) -> dict[str, str]:
    flattened: dict[str, str] = {}
    for key in sorted(params):
        value = params[key]
        dotted_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flattened.update(_flatten_params(value, prefix=dotted_key))
            continue
        flattened[dotted_key] = _format_param_value(value)
    return flattened


def _format_param_value(value: object) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (list, tuple)):
        return ",".join(_format_param_value(item) for item in value)
    return str(value)


def _render_box_border(kind: str, *, title: str | None = None) -> str:
    inner_width = _BANNER_WIDTH - 2
    if kind == "top":
        return f"╔{_render_border_title(inner_width=inner_width, title=title)}╗"
    if kind == "section":
        return f"╠{_render_border_title(inner_width=inner_width, title=title)}╣"
    if kind == "bottom":
        return f"╚{'═' * inner_width}╝"
    raise ValueError(f"Unsupported box border kind: {kind!r}")


def _render_border_title(*, inner_width: int, title: str | None) -> str:
    if not title:
        return "═" * inner_width
    decorated = f" {title} "
    return decorated.center(inner_width, "═")


def _render_box_lines(lines: list[str]) -> list[str]:
    wrapped_lines: list[str] = []
    content_width = _BANNER_WIDTH - 4
    for line in lines:
        chunks = textwrap.wrap(
            line,
            width=content_width,
            break_long_words=False,
            break_on_hyphens=False,
        )
        if not chunks:
            chunks = [""]
        for chunk in chunks:
            wrapped_lines.append(f"║ {chunk.ljust(content_width)} ║")
    return wrapped_lines


def _resolve_git_metadata(*, repo_root: Path) -> tuple[str, str]:
    sha_result = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    dirty_result = subprocess.run(
        ["git", "-C", str(repo_root), "status", "--porcelain"],
        check=False,
        capture_output=True,
        text=True,
    )

    if sha_result.returncode != 0 or dirty_result.returncode != 0:
        return "UNKNOWN", "UNKNOWN"

    sha = sha_result.stdout.strip() or "UNKNOWN"
    dirty = "true" if dirty_result.stdout.strip() else "false"
    return sha, dirty
