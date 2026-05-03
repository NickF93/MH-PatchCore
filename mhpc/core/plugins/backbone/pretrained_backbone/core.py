"""Core behavior for the unified `pretrained_backbone` plugin."""

from __future__ import annotations

from collections.abc import Callable

import torch
import torchvision.models as _models  # type: ignore[import-untyped]

from ..contracts import BackboneFeatureExtractor, BackboneSpec
from .feature_extractor import FeatureExtractor
from .layer_resolution import resolve_feature_layers
from .naming import normalize_backbone_name

_SUPPORTED_BACKBONES: tuple[str, ...] = ("resnetv2_50_21k", "wideresnet50")


def _create_timm_model(model_name: str, *, pretrained: bool) -> torch.nn.Module:
    import timm as _timm  # type: ignore[import-untyped]

    return _timm.create_model(model_name, pretrained=pretrained)


def _build_resnetv2_50_21k(device: torch.device) -> torch.nn.Module:
    return _create_timm_model("resnetv2_50x3_bitm_in21k", pretrained=True).to(device)


def _build_wideresnet50(device: torch.device) -> torch.nn.Module:
    return _models.wide_resnet50_2(weights="DEFAULT").to(device)


_BACKBONE_BUILDERS: dict[str, Callable[[torch.device], torch.nn.Module]] = {
    "resnetv2_50_21k": _build_resnetv2_50_21k,
    "wideresnet50": _build_wideresnet50,
}


def initialize_backbone_and_layers(
    *,
    backbone: BackboneSpec,
    embedding_layers: list[str],
    device: torch.device,
) -> tuple[torch.nn.Module, list[str]]:
    """Initialize selected pretrained backbone and resolve embedding layers."""
    backbone_name = normalize_backbone_name(backbone)
    builder = _BACKBONE_BUILDERS.get(backbone_name)
    if builder is None:
        supported = ", ".join(_SUPPORTED_BACKBONES)
        raise ValueError(
            "Plugin 'pretrained_backbone' requires one of "
            f"{supported}; got backbone='{backbone_name}'."
        )
    backbone_module = builder(device)
    resolved_layers = resolve_feature_layers(
        model=backbone_module,
        layers=embedding_layers,
    )
    return backbone_module, resolved_layers


def create_extractor(
    *,
    backbone: torch.nn.Module,
    resolved_embedding_layers: list[str],
) -> BackboneFeatureExtractor:
    """Create feature extractor for resolved layers."""
    return FeatureExtractor(
        backbone,
        resolved_embedding_layers,
    )
