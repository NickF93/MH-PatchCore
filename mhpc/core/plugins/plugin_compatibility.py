"""Compatibility validation for selected runtime plugins."""

from __future__ import annotations

from collections.abc import Iterator

from mhpc.core.runtime_plugin_bundle_contract import RuntimePluginBundle
from .plugin_capability import read_plugin_capability_metadata

_ENFORCE_LOCALITY_FRONTIER = True


def _iter_runtime_plugins(runtime_bundle: RuntimePluginBundle) -> Iterator[tuple[str, object]]:
    model_bundle = runtime_bundle.model_plugin_bundle
    yield "dataloader", runtime_bundle.dataloader_plugin
    yield "backbone", model_bundle.backbone_plugin
    yield "patch_align", model_bundle.patch_align_plugin
    yield "preprocess", model_bundle.preprocess_plugin
    yield "feature_agg", model_bundle.feature_agg_plugin
    yield "proj1", model_bundle.proj1_plugin
    yield "transform", model_bundle.transform_plugin
    yield "proj2", model_bundle.proj2_plugin
    yield "mem_agg", model_bundle.mem_agg_plugin
    yield "materialize", model_bundle.materialize_plugin
    yield "distance", model_bundle.distance_plugin
    yield "scoring", model_bundle.scoring_plugin


def _iter_locality_aware_plugins(
    runtime_bundle: RuntimePluginBundle,
) -> Iterator[tuple[str, object]]:
    model_bundle = runtime_bundle.model_plugin_bundle
    yield "proj1", model_bundle.proj1_plugin
    yield "transform", model_bundle.transform_plugin
    yield "proj2", model_bundle.proj2_plugin
    yield "mem_agg", model_bundle.mem_agg_plugin
    yield "materialize", model_bundle.materialize_plugin
    yield "distance", model_bundle.distance_plugin
    yield "scoring", model_bundle.scoring_plugin


def validate_runtime_plugin_compatibility(
    *,
    runtime_bundle: RuntimePluginBundle,
    training_contract: str,
) -> None:
    """Validate runtime plugin compatibility for the configured training contract."""
    if _ENFORCE_LOCALITY_FRONTIER:
        _validate_locality_frontier(runtime_bundle=runtime_bundle)

    if training_contract != "STREAMING":
        return

    for slot_name, plugin in _iter_runtime_plugins(runtime_bundle):
        capability_metadata = read_plugin_capability_metadata(
            plugin,
            slot_name=slot_name,
        )
        if not capability_metadata.supports_streaming:
            raise ValueError(
                "Incompatible plugin for STREAMING mode: "
                f"slot='{slot_name}' plugin_type='{type(plugin).__name__}' "
                "supports_streaming=False"
            )

        if capability_metadata.requires_full_dataset:
            raise ValueError(
                "Incompatible plugin for STREAMING mode: "
                f"slot='{slot_name}' plugin_type='{type(plugin).__name__}' "
                "requires_full_dataset=True"
            )

def _validate_locality_frontier(
    *,
    runtime_bundle: RuntimePluginBundle,
) -> None:
    frontier_slot: str | None = None
    for slot_name, plugin in _iter_locality_aware_plugins(runtime_bundle):
        capability_metadata = read_plugin_capability_metadata(
            plugin,
            slot_name=slot_name,
        )
        if (
            capability_metadata.requires_locality_context
            and not capability_metadata.preserves_locality
        ):
            raise ValueError(
                "Invalid locality capability declaration: "
                f"slot='{slot_name}' plugin_type='{type(plugin).__name__}' "
                "requires_locality_context=True preserves_locality=False"
            )
        if frontier_slot is None:
            if capability_metadata.requires_locality_context:
                frontier_slot = slot_name
            continue
        if not capability_metadata.preserves_locality:
            raise ValueError(
                "Incompatible plugin after locality frontier: "
                f"frontier_slot='{frontier_slot}' slot='{slot_name}' "
                f"plugin_type='{type(plugin).__name__}' "
                "preserves_locality=False"
            )
