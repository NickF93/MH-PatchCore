"""IPCA feature-aggregation plugin."""

from __future__ import annotations

import torch

from ..contracts import (
    FeatureAggregatorPlugin,
    FeatureReductionStrategy,
    ReductionSelection,
)
from .param_binding import FeatureAggParamBindingMixin
from .runtime import FeatureAggRuntimeMixin
from .strategy import (
    BatchPCAReductionStrategy,
    StreamingIncrementalPCAReductionStrategy,
)


class IPCAFeatureAggregatorPlugin(
    FeatureAggRuntimeMixin,
    FeatureAggParamBindingMixin,
    FeatureAggregatorPlugin,
):
    """Feature aggregation plugin selecting PCA reducers."""

    supports_streaming: bool = True
    requires_full_dataset: bool = False

    def forward_embed_feature_aggregation(
        self,
        *,
        features: torch.Tensor,
        forward_modules: torch.nn.ModuleDict,
    ) -> torch.Tensor:
        return forward_modules["preadapt_aggregator"](features)

    def create_feature_reduction_strategy(
        self,
        *,
        selection: ReductionSelection,
    ) -> FeatureReductionStrategy:
        if selection.algorithm == "STREAMING":
            return StreamingIncrementalPCAReductionStrategy(selection.pca_variance_ratio)
        return BatchPCAReductionStrategy(selection.pca_variance_ratio)

    def requires_fit_state(
        self,
        *,
        selection: ReductionSelection,
    ) -> bool:
        _ = selection
        return True
