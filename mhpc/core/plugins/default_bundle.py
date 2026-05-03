"""Facade for constructing the default plugin bundle."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .backbone.contracts import BackbonePlugin
from .dataloader.contracts import DataLoaderPlugin
from .distance.contracts import DistancePlugin
from .materialize.contracts import MaterializationPlugin
from ..model_plugin_bundle_contract import ModelPluginBundle
from .feature_agg.contracts import FeatureAggregatorPlugin
from .mem_agg.contracts import MemoryAggregationPlugin
from .patch_align.contracts import PatchAlignPlugin
from .preprocess.contracts import PreprocessPlugin
from .proj1.contracts import Projector1Plugin
from .proj2.contracts import Projector2Plugin
from .scoring.contracts import ScoringPlugin
from .transform.contracts import TransformPlugin
from .plugin_capability import collect_selected_plugin_capabilities
from .plugin_discovery import discover_internal_plugins
from .plugin_selection import select_plugins


@dataclass(frozen=True)
class DefaultPluginBundle:
    """Default runtime plugin instances wired for current behavior."""

    dataloader_plugin: DataLoaderPlugin
    model_plugin_bundle: ModelPluginBundle


@dataclass(frozen=True)
class DefaultModelPluginBundle:
    """Default model-side plugin instances wired for current behavior."""

    backbone_plugin: BackbonePlugin
    patch_align_plugin: PatchAlignPlugin
    preprocess_plugin: PreprocessPlugin
    feature_agg_plugin: FeatureAggregatorPlugin
    proj1_plugin: Projector1Plugin
    transform_plugin: TransformPlugin
    proj2_plugin: Projector2Plugin
    mem_agg_plugin: MemoryAggregationPlugin
    materialize_plugin: MaterializationPlugin
    distance_plugin: DistancePlugin
    scoring_plugin: ScoringPlugin


_CANONICAL_RUNTIME_STAGES: tuple[str, ...] = (
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
)

_CANONICAL_MODEL_STAGES: tuple[str, ...] = tuple(
    stage_name for stage_name in _CANONICAL_RUNTIME_STAGES if stage_name != "dataloader"
)


def _require_selected_plugin(
    selected: dict[str, object],
    key: str,
    *,
    expected_type: type[Any],
) -> Any:
    plugin = selected.get(key)
    if plugin is None:
        raise KeyError(f"Missing required default plugin for stage '{key}'")
    if not isinstance(plugin, expected_type):
        raise TypeError(
            "Selected plugin does not satisfy expected contract: "
            f"stage='{key}' expected='{expected_type.__name__}' "
            f"actual='{type(plugin).__name__}'"
        )
    return plugin


def _resolve_selection_map(
    *,
    selection_map: Mapping[str, str] | None,
    stage_names: tuple[str, ...],
    surface_name: str,
) -> dict[str, str]:
    if selection_map is None:
        raise ValueError(
            f"{surface_name} is required and must include all canonical stages: "
            f"{', '.join(stage_names)}"
        )

    unknown_stages = sorted(set(selection_map.keys()) - set(stage_names))
    if unknown_stages:
        raise KeyError(
            "Unsupported plugin selection stage(s): "
            f"{', '.join(unknown_stages)}"
        )

    missing_stages = [
        stage_name
        for stage_name in stage_names
        if stage_name not in selection_map
    ]
    if missing_stages:
        raise KeyError(
            "Missing plugin selection for canonical stage(s): "
            f"{', '.join(missing_stages)}"
        )

    resolved: dict[str, str] = {}
    for stage_name in stage_names:
        plugin_name = selection_map[stage_name]
        if not isinstance(plugin_name, str):
            raise ValueError(
                "Plugin selection values must be strings; "
                f"got stage='{stage_name}' type={type(plugin_name).__name__}"
            )
        stripped_name = plugin_name.strip()
        if not stripped_name:
            raise ValueError(
                "Plugin selection values must be non-empty strings; "
                f"got stage='{stage_name}'"
            )
        resolved[stage_name] = stripped_name
    return resolved


def _resolve_runtime_selection_map(
    selection_map: Mapping[str, str] | None,
) -> dict[str, str]:
    return _resolve_selection_map(
        selection_map=selection_map,
        stage_names=_CANONICAL_RUNTIME_STAGES,
        surface_name="Runtime plugin selection map",
    )


def _resolve_model_selection_map(
    selection_map: Mapping[str, str] | None,
) -> dict[str, str]:
    return _resolve_selection_map(
        selection_map=selection_map,
        stage_names=_CANONICAL_MODEL_STAGES,
        surface_name="Model plugin selection map",
    )


def _build_model_bundle_from_selected(
    selected: dict[str, object],
) -> DefaultModelPluginBundle:
    return DefaultModelPluginBundle(
        backbone_plugin=_require_selected_plugin(
            selected,
            "backbone",
            expected_type=BackbonePlugin,
        ),
        patch_align_plugin=_require_selected_plugin(
            selected,
            "patch_align",
            expected_type=PatchAlignPlugin,
        ),
        preprocess_plugin=_require_selected_plugin(
            selected,
            "preprocess",
            expected_type=PreprocessPlugin,
        ),
        feature_agg_plugin=_require_selected_plugin(
            selected,
            "feature_agg",
            expected_type=FeatureAggregatorPlugin,
        ),
        proj1_plugin=_require_selected_plugin(
            selected,
            "proj1",
            expected_type=Projector1Plugin,
        ),
        transform_plugin=_require_selected_plugin(
            selected,
            "transform",
            expected_type=TransformPlugin,
        ),
        proj2_plugin=_require_selected_plugin(
            selected,
            "proj2",
            expected_type=Projector2Plugin,
        ),
        mem_agg_plugin=_require_selected_plugin(
            selected,
            "mem_agg",
            expected_type=MemoryAggregationPlugin,
        ),
        materialize_plugin=_require_selected_plugin(
            selected,
            "materialize",
            expected_type=MaterializationPlugin,
        ),
        distance_plugin=_require_selected_plugin(
            selected,
            "distance",
            expected_type=DistancePlugin,
        ),
        scoring_plugin=_require_selected_plugin(
            selected,
            "scoring",
            expected_type=ScoringPlugin,
        ),
    )


def build_default_model_plugin_bundle(
    selection_map: Mapping[str, str] | None = None,
) -> DefaultModelPluginBundle:
    """Build the model-side plugin bundle from an explicit stage selection map."""
    discovered = discover_internal_plugins()
    selected = select_plugins(
        discovered,
        _resolve_model_selection_map(selection_map),
        validate_capabilities=False,
    )
    model_bundle = _build_model_bundle_from_selected(selected)
    collect_selected_plugin_capabilities(selected)
    return model_bundle


def build_default_plugin_bundle(
    selection_map: Mapping[str, str] | None = None,
) -> DefaultPluginBundle:
    """Build the current default runtime plugin set."""
    discovered = discover_internal_plugins()
    public_selection = _resolve_runtime_selection_map(selection_map)
    selected = select_plugins(
        discovered,
        dict(public_selection),
        validate_capabilities=False,
    )
    model_bundle = _build_model_bundle_from_selected(selected)
    dataloader_plugin = _require_selected_plugin(
        selected,
        "dataloader",
        expected_type=DataLoaderPlugin,
    )
    collect_selected_plugin_capabilities(selected)

    return DefaultPluginBundle(
        dataloader_plugin=dataloader_plugin,
        model_plugin_bundle=model_bundle,
    )
