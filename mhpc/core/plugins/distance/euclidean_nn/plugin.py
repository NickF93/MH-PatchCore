"""`euclidean_nn` distance plugin implementation."""

from __future__ import annotations

import numpy as np

from ..contracts import (
    DistanceAnomalyScorer,
    DistancePlugin,
    DistanceQueryPayload,
)
from .param_binding import DistanceParamBindingMixin
from .runtime import NearestNeighbourScorer


class EuclideanNNDistancePlugin(DistanceParamBindingMixin, DistancePlugin):
    """Distance plugin for current Euclidean nearest-neighbour query behavior."""

    supports_train: bool = True
    supports_inference: bool = True
    supports_streaming: bool = True
    requires_full_dataset: bool = False
    requires_locality_context: bool = False
    preserves_locality: bool = False

    def create_anomaly_scorer(
        self,
        *,
        n_nearest_neighbours: int,
    ) -> DistanceAnomalyScorer:
        return NearestNeighbourScorer(
            n_nearest_neighbours=n_nearest_neighbours
        )

    def query(
        self,
        *,
        anomaly_scorer: DistanceAnomalyScorer,
        features: np.ndarray,
        locality_context: object | None = None,
    ) -> DistanceQueryPayload:
        del locality_context
        return anomaly_scorer.predict([features])
