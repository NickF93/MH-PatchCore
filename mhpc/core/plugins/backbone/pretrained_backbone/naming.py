"""Backbone naming helpers local to the unified pretrained backbone plugin."""

from __future__ import annotations

from ..contracts import BackboneSpec


def normalize_backbone_name(backbone: BackboneSpec) -> str:
    """Normalize configured backbone spec to canonical string key."""
    if not isinstance(backbone, str):
        raise TypeError(
            "Backbone spec must be a string token. "
            f"Received type={type(backbone).__name__}."
        )
    normalized = backbone.strip()
    if not normalized:
        raise ValueError("Backbone spec must be a non-empty string token.")
    return normalized
