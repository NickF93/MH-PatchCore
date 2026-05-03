"""Runtime validators for plugin capability metadata."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class PluginCapabilityMetadata:
    """Capability flags declared by one plugin implementation."""

    supports_streaming: bool
    requires_full_dataset: bool
    requires_locality_context: bool
    preserves_locality: bool


def read_plugin_capability_metadata(
    plugin: object,
    *,
    slot_name: str,
) -> PluginCapabilityMetadata:
    """Read and validate capability metadata from one plugin object."""

    supports_streaming = _read_required_bool_capability(
        plugin,
        capability_name="supports_streaming",
        slot_name=slot_name,
    )
    requires_full_dataset = _read_required_bool_capability(
        plugin,
        capability_name="requires_full_dataset",
        slot_name=slot_name,
    )
    requires_locality_context = _read_optional_bool_capability(
        plugin,
        capability_name="requires_locality_context",
        slot_name=slot_name,
        default_value=False,
    )
    preserves_locality = _read_optional_bool_capability(
        plugin,
        capability_name="preserves_locality",
        slot_name=slot_name,
        default_value=False,
    )
    return PluginCapabilityMetadata(
        supports_streaming=supports_streaming,
        requires_full_dataset=requires_full_dataset,
        requires_locality_context=requires_locality_context,
        preserves_locality=preserves_locality,
    )


def collect_selected_plugin_capabilities(
    selected_plugins: Mapping[str, object],
) -> dict[str, PluginCapabilityMetadata]:
    """Collect and validate capability metadata for all selected plugins."""

    return {
        slot_name: read_plugin_capability_metadata(
            plugin,
            slot_name=slot_name,
        )
        for slot_name, plugin in selected_plugins.items()
    }


def _read_required_bool_capability(
    plugin: object,
    *,
    capability_name: str,
    slot_name: str,
) -> bool:
    if not hasattr(plugin, capability_name):
        raise TypeError(
            "Plugin capability is missing: "
            f"slot='{slot_name}' capability='{capability_name}' "
            f"plugin_type='{type(plugin).__name__}'"
        )
    value = getattr(plugin, capability_name)
    if not isinstance(value, bool):
        raise TypeError(
            "Plugin capability must be boolean: "
            f"slot='{slot_name}' capability='{capability_name}' "
            f"type={type(value).__name__}"
        )
    return value


def _read_optional_bool_capability(
    plugin: object,
    *,
    capability_name: str,
    slot_name: str,
    default_value: bool,
) -> bool:
    if not hasattr(plugin, capability_name):
        return bool(default_value)
    value = getattr(plugin, capability_name)
    if not isinstance(value, bool):
        raise TypeError(
            "Plugin capability must be boolean: "
            f"slot='{slot_name}' capability='{capability_name}' "
            f"type={type(value).__name__}"
        )
    return value


__all__ = [
    "PluginCapabilityMetadata",
    "collect_selected_plugin_capabilities",
    "read_plugin_capability_metadata",
]
