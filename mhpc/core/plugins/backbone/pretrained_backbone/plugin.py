"""`pretrained_backbone` backbone plugin implementation."""

from __future__ import annotations

import torch

from ..contracts import (
    BackboneFeatureExtractor,
    BackbonePlugin,
)
from .core import (
    create_extractor,
    initialize_backbone_and_layers as initialize_backbone_and_layers_impl,
)
from .param_binding import PretrainedBackboneParamBindingMixin


class PretrainedBackbonePlugin(PretrainedBackboneParamBindingMixin, BackbonePlugin):
    """Unified plugin for pretrained backbone variants."""

    supports_streaming: bool = True
    requires_full_dataset: bool = False

    def initialize_backbone_and_layers(
        self,
        *,
        device: torch.device,
    ) -> tuple[torch.nn.Module, list[str]]:
        return initialize_backbone_and_layers_impl(
            backbone=self._backbone_params.backbone,
            embedding_layers=list(self._backbone_params.embedding_layers),
            device=device,
        )

    def create_feature_extractor(
        self,
        *,
        backbone: torch.nn.Module,
        resolved_embedding_layers: list[str],
    ) -> BackboneFeatureExtractor:
        return create_extractor(
            backbone=backbone,
            resolved_embedding_layers=resolved_embedding_layers,
        )
