"""Plugin-local preprocess runtime modules for ``pc_preprocess``."""

from __future__ import annotations

import torch
import torch.nn.functional as F


class MeanMapper(torch.nn.Module):
    def __init__(self, preprocessing_dim: int) -> None:
        super(MeanMapper, self).__init__()
        self.preprocessing_dim = preprocessing_dim

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        features = features.reshape(len(features), 1, -1)
        return F.adaptive_avg_pool1d(features, self.preprocessing_dim).squeeze(1)


class Preprocessing(torch.nn.Module):
    def __init__(
        self,
        input_dims: list[int] | tuple[int, ...],
        output_dim: int,
    ) -> None:
        super(Preprocessing, self).__init__()
        self.input_dims = input_dims
        self.output_dim = output_dim

        self.preprocessing_modules = torch.nn.ModuleList()
        for _input_dim in input_dims:
            module = MeanMapper(output_dim)
            self.preprocessing_modules.append(module)

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        mapped_features: list[torch.Tensor] = []
        for module, feature in zip(self.preprocessing_modules, features):
            mapped_features.append(module(feature))
        return torch.stack(mapped_features, dim=1)


__all__ = ["MeanMapper", "Preprocessing"]
