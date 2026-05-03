"""Plugin-local category-aware augmentation policy helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def deep_merge_augment_cfg(
    base_cfg: Mapping[str, Any],
    override_cfg: Mapping[str, Any],
) -> dict[str, Any]:
    """Deep-merge two augmentation mappings with override precedence."""
    merged: dict[str, Any] = _clone_mapping(base_cfg)
    for key, override_value in override_cfg.items():
        base_value = merged.get(key)
        if isinstance(base_value, Mapping) and isinstance(override_value, Mapping):
            merged[key] = deep_merge_augment_cfg(base_value, override_value)
            continue
        merged[key] = _clone_value(override_value)
    return merged


def validate_category_overrides(
    *,
    available_categories: Sequence[str],
    overrides: Mapping[str, Mapping[str, Any]],
) -> None:
    """Validate that override categories exist in the configured category set."""
    allowed_categories = set(available_categories)
    unknown_categories: list[str] = []
    for category, override in overrides.items():
        if not isinstance(category, str) or not category.strip():
            raise ValueError(
                "data.train_augment_overrides keys must be non-empty strings"
            )
        if not isinstance(override, Mapping):
            raise ValueError(
                f"data.train_augment_overrides.{category} must be a mapping"
            )
        if category not in allowed_categories:
            unknown_categories.append(category)
    if unknown_categories:
        raise ValueError(
            "data.train_augment_overrides contains unknown categories: "
            f"{', '.join(sorted(set(unknown_categories)))}"
        )


def resolve_category_augment_cfg(
    *,
    base_cfg: Mapping[str, Any],
    overrides: Mapping[str, Mapping[str, Any]],
    category: str,
    available_categories: Sequence[str],
) -> dict[str, Any]:
    """Resolve effective train augmentation config for one dataset category."""
    validate_category_overrides(
        available_categories=available_categories,
        overrides=overrides,
    )
    category_override = overrides.get(category, {})
    return deep_merge_augment_cfg(base_cfg, category_override)


def _clone_mapping(mapping: Mapping[str, Any]) -> dict[str, Any]:
    cloned: dict[str, Any] = {}
    for key, value in mapping.items():
        if not isinstance(key, str):
            raise TypeError(
                "Augmentation configuration keys must be strings; "
                f"got {type(key).__name__}."
            )
        cloned[key] = _clone_value(value)
    return cloned


def _clone_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _clone_mapping(value)
    if isinstance(value, list):
        return [_clone_value(item) for item in value]
    return value

