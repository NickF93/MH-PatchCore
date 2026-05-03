"""Shared repository-root path resolution helpers."""

from __future__ import annotations

from pathlib import Path

_REPO_ROOT_REQUIRED_DIRS = ("configs", "mhpc", "scripts")
_REPO_ROOT_REQUIRED_FILES = ("pyproject.toml", "README.md")


def resolve_repo_root(anchor: str | Path) -> Path:
    """Resolve the repository root from any file or directory inside the repo.

    The repository root is identified by the canonical top-level
    source/config/script directories and package metadata. This keeps
    repo-relative path resolution deterministic without duplicating brittle
    ``parents[...]`` calculations across modules.
    """

    anchor_path = Path(anchor).expanduser().resolve()
    search_root = anchor_path if anchor_path.is_dir() else anchor_path.parent

    for candidate in (search_root, *search_root.parents):
        if _is_repo_root(candidate):
            return candidate

    raise RuntimeError(
        "Unable to resolve repository root from anchor: "
        f"{anchor_path}"
    )


def _is_repo_root(candidate: Path) -> bool:
    return all((candidate / dirname).is_dir() for dirname in _REPO_ROOT_REQUIRED_DIRS) and all(
        (candidate / filename).is_file() for filename in _REPO_ROOT_REQUIRED_FILES
    )
