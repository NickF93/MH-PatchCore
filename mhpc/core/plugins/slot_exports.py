"""Generic slot-root export helpers for plugin class symbols.

These helpers keep slot package ``__init__`` modules free from concrete plugin-id
hardcoding by resolving exports from discovered plugin package metadata.
"""

from __future__ import annotations

from importlib import import_module
from pathlib import Path


def build_slot_export_map(*, slot_package: str, slot_dir: Path) -> dict[str, tuple[str, str]]:
    """Build ``export_name -> (module_name, export_name)`` from plugin ``__all__``."""
    normalized_slot_package = (
        slot_package[:-9]
        if slot_package.endswith(".__init__")
        else slot_package
    )
    export_map: dict[str, tuple[str, str]] = {}
    for plugin_dir in sorted(slot_dir.iterdir()):
        if not plugin_dir.is_dir():
            continue
        if plugin_dir.name in {"common", "__pycache__"}:
            continue
        plugin_init = plugin_dir / "__init__.py"
        if not plugin_init.is_file():
            continue
        module_name = f"{normalized_slot_package}.{plugin_dir.name}"
        module = import_module(module_name)
        exports = getattr(module, "__all__", None)
        if not isinstance(exports, list):
            continue
        for export_name in exports:
            if not isinstance(export_name, str):
                continue
            export_obj = getattr(module, export_name, None)
            if not isinstance(export_obj, type):
                continue
            if export_name in export_map:
                existing_module, _ = export_map[export_name]
                raise ValueError(
                    "Duplicate plugin export name across slot packages: "
                    f"slot='{normalized_slot_package}' export='{export_name}' "
                    f"modules='{existing_module}' and '{module_name}'"
                )
            export_map[export_name] = (module_name, export_name)
    return export_map


def resolve_slot_export(
    name: str,
    *,
    export_map: dict[str, tuple[str, str]],
) -> type[object]:
    """Resolve one class export from a pre-built slot export map."""
    target = export_map.get(name)
    if target is None:
        raise AttributeError(name)
    module_name, export_name = target
    module = import_module(module_name)
    export_obj = getattr(module, export_name, None)
    if not isinstance(export_obj, type):
        raise TypeError(
            "Resolved slot export is not a class: "
            f"module='{module_name}' export='{export_name}'"
        )
    return export_obj


__all__ = [
    "build_slot_export_map",
    "resolve_slot_export",
]
