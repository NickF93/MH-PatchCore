"""Factory helpers for memory-aggregation runtime state creation."""

from __future__ import annotations

from .contracts import (
    AggregationRuntimeState,
    MemAggRuntimeContext,
    MemoryAggregationPlugin,
)
from ..plugin_discovery import discover_internal_plugins


def _discover_memory_aggregation_plugins() -> dict[str, MemoryAggregationPlugin]:
    discovered = discover_internal_plugins()
    stage_plugins = discovered.get("mem_agg")
    if stage_plugins is None:
        raise RuntimeError("Missing discovered plugin stage 'mem_agg'.")

    resolved: dict[str, MemoryAggregationPlugin] = {}
    for plugin_id, plugin in stage_plugins.items():
        if not isinstance(plugin, MemoryAggregationPlugin):
            raise TypeError(
                "Discovered mem_agg plugin does not satisfy slot contract: "
                f"plugin_id='{plugin_id}' actual='{type(plugin).__name__}'"
            )
        resolved[plugin_id] = plugin
    return resolved


def create_memory_aggregation_state_for_plugin(
    *,
    plugin_id: str,
    runtime_context: MemAggRuntimeContext,
) -> AggregationRuntimeState:
    """Create memory aggregation runtime state for one concrete plugin id."""
    plugins = _discover_memory_aggregation_plugins()
    plugin = plugins.get(plugin_id)
    if plugin is None:
        raise ValueError(f"Unsupported memory_agg plugin id '{plugin_id}'.")
    return plugin.create_runtime_state(runtime_context=runtime_context)
