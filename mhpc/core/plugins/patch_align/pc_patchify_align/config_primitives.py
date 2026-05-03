"""Plugin-local parser primitives for `patch_align:pc_patchify_align`."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def _require_int(obj: Mapping[str, Any], key: str) -> int:
    value = obj.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    return value
