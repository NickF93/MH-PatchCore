"""Plugin-local parser primitives for `backbone:pretrained_backbone`."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def _require_non_empty_string(obj: Mapping[str, Any], key: str) -> str:
    value = obj.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value.strip()


def _require_string_list(obj: Mapping[str, Any], key: str) -> list[str]:
    raw = obj.get(key)
    if not isinstance(raw, list):
        raise ValueError(f"{key} must be a list of strings")
    out: list[str] = []
    for index, item in enumerate(raw):
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{key}[{index}] must be a non-empty string")
        out.append(item.strip())
    return out
