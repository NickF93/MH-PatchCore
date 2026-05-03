"""Minimal plugin selection over discovered plugins."""

from __future__ import annotations

from .plugin_discovery import DiscoveredPlugins
from .plugin_capability import read_plugin_capability_metadata

PluginSelection = dict[str, str]
SelectedPlugins = dict[str, object]


def select_plugins(
    discovered_plugins: DiscoveredPlugins,
    selection: PluginSelection,
    *,
    validate_capabilities: bool = True,
) -> SelectedPlugins:
    """Select exactly one plugin per requested stage."""
    selected: SelectedPlugins = {}
    for stage_name, plugin_name in selection.items():
        stage_plugins = discovered_plugins.get(stage_name)
        if stage_plugins is None:
            raise KeyError(f"Missing discovered stage: '{stage_name}'")

        plugin = stage_plugins.get(plugin_name)
        if plugin is None:
            raise KeyError(
                "Missing plugin selection for stage "
                f"'{stage_name}' and plugin '{plugin_name}'"
            )
        if validate_capabilities:
            read_plugin_capability_metadata(
                plugin,
                slot_name=stage_name,
            )
        selected[stage_name] = plugin

    return selected
