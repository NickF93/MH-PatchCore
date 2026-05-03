"""Shared bind-once parameter surfaces for slot plugins."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from mhpc.util.repo_paths import resolve_repo_root

CanonicalTrainingContract = Literal["OFFLINE", "STREAMING"]


def normalize_training_contract(value: object) -> CanonicalTrainingContract:
    """Normalize canonical training contract token used in plugin binding."""
    raw_value = value.name if hasattr(value, "name") else value
    if not isinstance(raw_value, str):
        raise TypeError(
            "training_contract must be a string token: "
            f"type={type(raw_value).__name__}"
        )
    normalized = raw_value.strip().upper()
    if normalized not in {"OFFLINE", "STREAMING"}:
        raise ValueError(
            "training_contract must be one of {'OFFLINE', 'STREAMING'}: "
            f"value={raw_value!r}"
        )
    return normalized  # type: ignore[return-value]


def normalize_reproducibility_seed(value: object) -> int:
    """Validate canonical run-wide reproducibility seed used at bind time."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(
            "bind_context.seed must be an integer: "
            f"type={type(value).__name__}"
        )
    return int(value)


def normalize_repo_root(value: object) -> Path:
    """Validate canonical repository root delivered at bind time."""
    if isinstance(value, Path):
        return value.expanduser().resolve()
    if isinstance(value, str) and value.strip():
        return Path(value).expanduser().resolve()
    raise TypeError(
        "bind_context.repo_root must be a path-like value: "
        f"type={type(value).__name__}"
    )


@dataclass(frozen=True)
class PluginBindContext:
    """Typed bind-time metadata shared across plugin slots."""

    training_contract: CanonicalTrainingContract
    seed: int
    repo_root: Path


def build_plugin_bind_context(
    *,
    training_contract: object,
    seed: object,
    repo_root: object | None = None,
) -> PluginBindContext:
    """Build canonical plugin bind metadata from orchestration inputs."""
    return PluginBindContext(
        training_contract=normalize_training_contract(training_contract),
        seed=normalize_reproducibility_seed(seed),
        repo_root=normalize_repo_root(
            resolve_repo_root(__file__) if repo_root is None else repo_root
        ),
    )


class PluginParamBindingMixin:
    """Default bind-once implementation for slot plugin classes."""

    _bound_params: dict[str, Any]
    _bound_bind_context: PluginBindContext

    def bind_params(
        self,
        *,
        params: Mapping[str, Any],
        bind_context: PluginBindContext,
    ) -> None:
        """Persist one canonical slot-parameter bundle for this plugin instance."""
        if not isinstance(params, Mapping):
            raise TypeError(
                "params must be a mapping for plugin bind_params: "
                f"type={type(params).__name__}"
            )
        if not isinstance(bind_context, PluginBindContext):
            raise TypeError(
                "bind_context must be PluginBindContext for plugin bind_params: "
                f"type={type(bind_context).__name__}"
            )
        self._bound_params = dict(params)
        self._bound_bind_context = bind_context
