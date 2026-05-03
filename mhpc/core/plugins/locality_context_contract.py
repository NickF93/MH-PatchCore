"""Generic aligned-locality context contract for downstream slot plumbing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class LocalityContext:
    """Explicit aligned patch-grid metadata for locality-preserving slot flows."""

    batch_size: int
    patch_shape: tuple[int, int]
    flatten_order: Literal["image_major_row_major"] = "image_major_row_major"


__all__ = ["LocalityContext"]
