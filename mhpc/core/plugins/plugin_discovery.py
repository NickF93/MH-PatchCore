"""Autodiscovery for built-in plugins under authoritative pipeline slots."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Iterator

from mhpc.core.pipeline_stage_contract import canonical_slot_stages

DiscoveredPlugins = dict[str, dict[str, object]]

_CANONICAL_SLOT_STAGES: tuple[str, ...] = canonical_slot_stages()


def _resolve_default_export(module_name: str) -> str:
    module = import_module(module_name)
    exports = getattr(module, "__all__", None)
    if not isinstance(exports, list):
        raise ValueError(
            "Default plugin module __all__ must be a list: "
            f"{module_name}"
        )
    if len(exports) != 1:
        raise ValueError(
            "Default plugin module __all__ must contain exactly one export: "
            f"{module_name}"
        )
    export_name = exports[0]
    if not isinstance(export_name, str):
        raise TypeError(
            "Default plugin module __all__ entry must be a string: "
            f"{module_name}"
        )
    return export_name


def _assert_no_unexpected_stage_directories(plugins_dir: Path) -> None:
    allowed_stage_names = set(_CANONICAL_SLOT_STAGES)
    for stage_dir in sorted(plugins_dir.iterdir()):
        if not stage_dir.is_dir():
            continue
        if stage_dir.name == "__pycache__":
            continue
        if stage_dir.name in allowed_stage_names:
            continue
        raise ValueError(
            "Unexpected plugin stage directory detected: "
            f"'{stage_dir.name}'. Allowed stages: {sorted(allowed_stage_names)}"
        )


def _iter_plugin_candidate_modules(
    plugins_dir: Path,
    *,
    package_name: str | None = None,
) -> Iterator[tuple[str, str, str]]:
    resolved_package_name = __package__ if package_name is None else package_name
    if not resolved_package_name:
        raise ValueError("Plugin discovery package name must be non-empty.")

    for stage_name in _CANONICAL_SLOT_STAGES:
        stage_dir = plugins_dir / stage_name
        if not stage_dir.is_dir():
            raise ValueError(
                "Missing authoritative plugin stage directory: "
                f"'{stage_name}'"
            )
        has_plugin = False
        for plugin_dir in sorted(stage_dir.iterdir()):
            if not plugin_dir.is_dir():
                continue
            if plugin_dir.name in {"__pycache__", "common"}:
                continue
            has_plugin = True
            plugin_init = plugin_dir / "__init__.py"
            if not plugin_init.is_file():
                raise ValueError(
                    "Missing plugin entrypoint for discovered plugin: "
                    f"stage='{stage_name}' plugin_id='{plugin_dir.name}' "
                    f"path='{plugin_init}'"
                )
            module_name = f"{resolved_package_name}.{stage_name}.{plugin_dir.name}"
            yield stage_name, plugin_dir.name, module_name
        if not has_plugin:
            raise ValueError(
                "Missing plugin implementations for authoritative stage: "
                f"stage='{stage_name}'"
            )


def _register_discovered_plugin(
    discovered: DiscoveredPlugins,
    *,
    stage_name: str,
    plugin_id: str,
    plugin_instance: object,
) -> None:
    stage_plugins = discovered.setdefault(stage_name, {})
    if plugin_id in stage_plugins:
        raise ValueError(
            "Duplicate plugin registration detected for "
            f"stage='{stage_name}' plugin_id='{plugin_id}'"
        )
    stage_plugins[plugin_id] = plugin_instance


def discover_internal_plugins(
    *,
    plugins_dir: Path | None = None,
    package_name: str | None = None,
) -> DiscoveredPlugins:
    """Discover and instantiate built-in plugins under all internal slots."""
    resolved_plugins_dir = (
        Path(__file__).resolve().parent if plugins_dir is None else plugins_dir
    )
    resolved_package_name = __package__ if package_name is None else package_name
    _assert_no_unexpected_stage_directories(resolved_plugins_dir)
    discovered: DiscoveredPlugins = {}
    for stage_name, plugin_id, module_name in _iter_plugin_candidate_modules(
        resolved_plugins_dir,
        package_name=resolved_package_name,
    ):
        export_name = _resolve_default_export(module_name)
        plugin_class = getattr(import_module(module_name), export_name, None)
        if not isinstance(plugin_class, type):
            raise TypeError(
                "Discovered default export is not a class: "
                f"{module_name}.{export_name}"
            )
        _register_discovered_plugin(
            discovered,
            stage_name=stage_name,
            plugin_id=plugin_id,
            plugin_instance=plugin_class(),
        )

    return discovered


def discover_default_plugins() -> DiscoveredPlugins:
    """Discover and expose built-in plugins on canonical public slots only."""
    return discover_internal_plugins()
