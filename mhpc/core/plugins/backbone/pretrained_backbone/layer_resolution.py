"""Layer-resolution helpers local to the unified pretrained backbone plugin."""

from __future__ import annotations

import torch

_LAYER_ALIASES_BY_BACKBONE_CLASS: dict[str, dict[str, str]] = {
    "ResNetV2": {
        "layer1": "stages.0",
        "layer2": "stages.1",
        "layer3": "stages.2",
        "layer4": "stages.3",
    },
}


def resolve_feature_layers(
    *,
    model: torch.nn.Module,
    layers: list[str],
) -> list[str]:
    """Resolve canonical layer aliases to model-specific module names."""
    available_layers = {name for name, _ in model.named_modules()}
    if all(layer in available_layers for layer in layers):
        return list(layers)

    alias_map = _LAYER_ALIASES_BY_BACKBONE_CLASS.get(model.__class__.__name__, {})
    resolved = [alias_map.get(layer, layer) for layer in layers]
    if all(layer in available_layers for layer in resolved):
        return resolved

    unresolved = [layer for layer in resolved if layer not in available_layers]
    available_preview = ", ".join(sorted(list(available_layers))[:30])
    raise ValueError(
        "Unable to resolve requested embedding layers against backbone modules. "
        f"requested={layers}, resolved={resolved}, missing={unresolved}, "
        f"available_sample=[{available_preview}]"
    )
