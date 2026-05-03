"""PatchCore preprocess plugin implementation."""

from __future__ import annotations

import torch

from ..contracts import PreprocessPlugin
from .runtime import Preprocessing
from .param_binding import PCPreprocessParamBindingMixin


class PCPreprocessPlugin(PCPreprocessParamBindingMixin, PreprocessPlugin):
    """Behavior-preserving plugin for preprocessing forward wiring."""

    supports_streaming: bool = True
    requires_full_dataset: bool = False

    def forward_embed_preprocess(
        self,
        *,
        features: list[torch.Tensor],
        forward_modules: torch.nn.ModuleDict,
    ) -> torch.Tensor:
        return forward_modules["preprocessing"](features)

    def create_preprocessing_module(
        self,
        *,
        input_dims: list[int] | tuple[int, ...],
    ) -> torch.nn.Module:
        return Preprocessing(
            input_dims=input_dims,
            output_dim=self._preprocess_params.pretrain_embed_dimension,
        )
