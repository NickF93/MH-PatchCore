"""Plugin-local runtime modules for `feature_agg:ipca`."""

from __future__ import annotations

from typing import Protocol, cast

import torch
import torch.nn.functional as F


class Aggregator(torch.nn.Module):
    def __init__(self, target_dim: int) -> None:
        super().__init__()
        self.target_dim = target_dim

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        features = features.reshape(len(features), 1, -1)
        features = F.adaptive_avg_pool1d(features, self.target_dim)
        return features.reshape(len(features), -1)


class _TargetEmbedDimensionResolver(Protocol):
    def resolve_target_embed_dimension(self) -> int: ...


class FeatureAggRuntimeMixin:
    """Feature_agg runtime-module factory localized to `ipca`."""

    def create_preadapt_aggregator_module(self) -> torch.nn.Module:
        resolver = cast(_TargetEmbedDimensionResolver, self)
        return Aggregator(target_dim=int(resolver.resolve_target_embed_dimension()))
