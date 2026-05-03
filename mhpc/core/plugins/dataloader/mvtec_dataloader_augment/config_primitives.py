"""Plugin-local parser primitives for `dataloader:mvtec_dataloader_augment`."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def _require_optional_mapping(
    obj: Mapping[str, Any],
    key: str,
) -> Mapping[str, Any] | None:
    if key not in obj:
        return None
    value = obj.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be a mapping")
    return value


def _require_bool(obj: Mapping[str, Any], key: str) -> bool:
    value = obj.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be a boolean")
    return value


def _require_optional_bool(obj: Mapping[str, Any], key: str, default: bool) -> bool:
    if key not in obj:
        return default
    value = obj.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be a boolean")
    return value


def _require_optional_string(obj: Mapping[str, Any], key: str, default: str) -> str:
    if key not in obj:
        return default
    value = obj.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value.strip()


def _require_int(obj: Mapping[str, Any], key: str) -> int:
    value = obj.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    return value


def _require_optional_float(obj: Mapping[str, Any], key: str, default: float) -> float:
    if key not in obj:
        return default
    value = obj.get(key)
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError(f"{key} must be a float")
    return float(value)


def _require_optional_int_list(
    obj: Mapping[str, Any],
    key: str,
    default: tuple[int, ...],
) -> tuple[int, ...]:
    if key not in obj:
        return tuple(default)
    value = obj.get(key)
    if not isinstance(value, list):
        raise ValueError(f"{key} must be a list of integers")
    parsed: list[int] = []
    for idx, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, int):
            raise ValueError(f"{key}[{idx}] must be an integer")
        parsed.append(item)
    return tuple(parsed)


def _require_non_empty_string(obj: Mapping[str, Any], key: str) -> str:
    value = obj.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value.strip()


def _require_int_pair(obj: Mapping[str, Any], key: str) -> tuple[int, int]:
    raw = obj.get(key)
    if not isinstance(raw, list) or len(raw) != 2:
        raise ValueError(f"{key} must be a list with two integers")
    first, second = raw
    if isinstance(first, bool) or not isinstance(first, int):
        raise ValueError(f"{key}[0] must be an integer")
    if isinstance(second, bool) or not isinstance(second, int):
        raise ValueError(f"{key}[1] must be an integer")
    return first, second
