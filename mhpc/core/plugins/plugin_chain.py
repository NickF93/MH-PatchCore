"""Runtime plugin-chain facade for startup resolution."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from mhpc.core.model_plugin_bundle_contract import ModelPluginBundle
from mhpc.core.pipeline_stage_contract import canonical_slot_stages
from mhpc.core.runtime_plugin_bundle_contract import RuntimePluginBundle
from mhpc.util.param_binding import PluginBindContext

from .default_bundle import build_default_plugin_bundle
from .dataloader.contracts import DataLoaderPlugin
from .plugin_compatibility import validate_runtime_plugin_compatibility


@dataclass(frozen=True)
class DefaultRuntimePluginChain:
    """Runtime plugin chain resolved once at startup."""

    dataloader_plugin: DataLoaderPlugin
    model_plugin_bundle: ModelPluginBundle


def _selected_plugins_map(runtime_bundle: RuntimePluginBundle) -> dict[str, object]:
    model_bundle = runtime_bundle.model_plugin_bundle
    return {
        "dataloader": runtime_bundle.dataloader_plugin,
        "backbone": model_bundle.backbone_plugin,
        "patch_align": model_bundle.patch_align_plugin,
        "preprocess": model_bundle.preprocess_plugin,
        "feature_agg": model_bundle.feature_agg_plugin,
        "proj1": model_bundle.proj1_plugin,
        "transform": model_bundle.transform_plugin,
        "proj2": model_bundle.proj2_plugin,
        "mem_agg": model_bundle.mem_agg_plugin,
        "materialize": model_bundle.materialize_plugin,
        "distance": model_bundle.distance_plugin,
        "scoring": model_bundle.scoring_plugin,
    }


def _bind_selected_plugin_params(
    *,
    runtime_bundle: RuntimePluginBundle,
    slot_params_map: Mapping[str, Mapping[str, Any]],
    bind_context: PluginBindContext,
) -> None:
    selected_plugins = _selected_plugins_map(runtime_bundle)
    canonical_slots = canonical_slot_stages()
    missing_param_slots = [
        slot_name
        for slot_name in canonical_slots
        if slot_name not in slot_params_map
    ]
    if missing_param_slots:
        raise KeyError(
            "Missing slot parameter bundle(s) for runtime plugin bind: "
            f"{', '.join(missing_param_slots)}"
        )
    for slot_name in canonical_slots:
        plugin = selected_plugins[slot_name]
        raw_params = slot_params_map[slot_name]
        bind_params = getattr(plugin, "bind_params", None)
        if not callable(bind_params):
            raise TypeError(
                "Selected plugin does not expose required bind_params hook: "
                f"slot='{slot_name}' plugin_type='{type(plugin).__name__}'"
            )
        bind_params(
            params=raw_params,
            bind_context=bind_context,
        )


def build_runtime_plugin_chain(
    *,
    selection_map: Mapping[str, str],
    bind_context: PluginBindContext,
    slot_params_map: Mapping[str, Mapping[str, Any]],
) -> RuntimePluginBundle:
    """Resolve discover->select->compatibility into one runtime chain object."""
    runtime_bundle = build_default_plugin_bundle(selection_map=selection_map)
    _bind_selected_plugin_params(
        runtime_bundle=runtime_bundle,
        slot_params_map=slot_params_map,
        bind_context=bind_context,
    )
    validate_runtime_plugin_compatibility(
        runtime_bundle=runtime_bundle,
        training_contract=bind_context.training_contract,
    )
    return DefaultRuntimePluginChain(
        dataloader_plugin=runtime_bundle.dataloader_plugin,
        model_plugin_bundle=runtime_bundle.model_plugin_bundle,
    )
