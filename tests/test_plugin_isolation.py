from __future__ import annotations

import ast
from pathlib import Path


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def test_concrete_plugins_do_not_import_peer_plugins() -> None:
    plugin_root = Path("mhpc/core/plugins")
    concrete_plugin_files = [
        path
        for path in plugin_root.glob("*/*/*.py")
        if path.parts[-1] != "__init__.py"
    ]
    assert concrete_plugin_files

    violations: list[str] = []
    for path in concrete_plugin_files:
        slot = path.parts[3]
        plugin_id = path.parts[4]
        prefix = f"mhpc.core.plugins.{slot}."
        for imported in _imports(path):
            if not imported.startswith(prefix):
                continue
            imported_parts = imported.split(".")
            if len(imported_parts) >= 5 and imported_parts[4] != plugin_id:
                violations.append(f"{path}: {imported}")

    assert violations == []


def test_materialize_plugins_are_not_backed_by_shared_plugin_code() -> None:
    common_dir = Path("mhpc/core/plugins/materialize/common")
    assert not any(common_dir.glob("*.py"))
