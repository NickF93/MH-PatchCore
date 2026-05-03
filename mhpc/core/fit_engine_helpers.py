"""Generic helper utilities for FitEngine orchestration seams."""

from __future__ import annotations

from typing import Any, Protocol, cast


class _SelectionFactory(Protocol):
    """Callable protocol for stage factories that accept one keyword payload."""

    def __call__(self, **kwargs: Any) -> Any:
        ...


def _resolve_plugin_factory(
    *,
    plugin: Any,
    factory_name: str,
) -> _SelectionFactory:
    """Resolve and validate a single-argument plugin factory by name."""
    factory = getattr(plugin, factory_name, None)
    if factory is None:
        raise AttributeError(
            "Plugin is missing required factory method: "
            f"{factory_name!r}"
        )
    if not callable(factory):
        raise TypeError(
            "Plugin factory attribute is not callable: "
            f"{factory_name!r}"
        )
    return cast(_SelectionFactory, factory)


def _create_stage_runtime_state_impl(
    *,
    plugin: Any,
    factory_name: str,
    argument_name: str,
    argument_value: Any,
) -> Any:
    """Create stage runtime state via a validated plugin factory call."""
    factory = _resolve_plugin_factory(
        plugin=plugin,
        factory_name=factory_name,
    )
    return factory(**{argument_name: argument_value})
