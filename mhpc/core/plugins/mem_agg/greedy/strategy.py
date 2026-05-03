"""Greedy coreset memory aggregation strategy."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from ..contracts import AggregationRuntimeMetadata
from .sampler import ApproximateGreedyCoresetSampler


class GreedyCoresetAggregationStrategy:
    """Greedy coreset aggregation (collect-all then sample)."""

    def __init__(self, percentage: float, device: torch.device) -> None:
        self._batches: list[np.ndarray] = []
        self._sampler = ApproximateGreedyCoresetSampler(
            percentage=percentage,
            device=device,
        )

    @property
    def name(self) -> str:
        return "GREEDY"

    @property
    def is_streaming(self) -> bool:
        return False

    @property
    def supports_multi_pass(self) -> bool:
        return False

    def update(
        self,
        batch: np.ndarray,
        locality_context: object | None = None,
        update_context: object | None = None,
    ) -> None:
        _ = update_context
        del locality_context
        self._batches.append(np.asarray(batch))

    def get_centroids(self) -> np.ndarray:
        if not self._batches:
            raise RuntimeError("GreedyCoresetAggregationStrategy received no batches.")
        features = np.concatenate(self._batches, axis=0)
        return np.asarray(self._sampler.run(features))

    def export_state(self) -> dict[str, Any]:
        return {}

    def runtime_metadata(self) -> AggregationRuntimeMetadata:
        return AggregationRuntimeMetadata()
