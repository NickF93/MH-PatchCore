"""`kmeans` memory aggregation plugin implementation."""

from __future__ import annotations

from ..contracts import (
    AggregationRuntimeState,
    MemAggRuntimeContext,
    MemoryAggregationPlugin,
)
from .param_binding import MemoryAggParamBindingMixin
from .strategy import (
    BatchKMeansAggregationStrategy,
    StreamingMiniBatchKMeansAggregationStrategy,
)


class KMeansMemoryAggregationPlugin(MemoryAggParamBindingMixin, MemoryAggregationPlugin):
    """Memory aggregation plugin for `KMEANS` aggregation."""

    supports_streaming: bool = True
    requires_full_dataset: bool = False
    requires_locality_context: bool = False
    preserves_locality: bool = False

    def create_runtime_state(
        self,
        *,
        runtime_context: MemAggRuntimeContext,
    ) -> AggregationRuntimeState:
        resolved_n_clusters = self._resolve_n_clusters(
            feature_count=runtime_context.feature_count
        )
        if resolved_n_clusters is None:
            raise ValueError(
                "KMEANS aggregation requires resolved n_clusters in strategy selection."
            )
        params = self._mem_agg_params
        seed = int(self._bound_bind_context.seed)
        if runtime_context.training_contract.strip().upper() == "STREAMING":
            return StreamingMiniBatchKMeansAggregationStrategy(
                n_clusters=resolved_n_clusters,
                minibatch_size=256,
                random_state=seed,
                enforce_reference_limit=params.streaming_enforce_cluster_budget,
            )
        return BatchKMeansAggregationStrategy(
            n_clusters=resolved_n_clusters,
            random_state=seed,
        )
