"""`kcenter` memory aggregation plugin implementation."""

from __future__ import annotations

from ..contracts import (
    AggregationRuntimeState,
    MemAggRuntimeContext,
    MemoryAggregationPlugin,
)
from .param_binding import MemoryAggParamBindingMixin
from .strategy import StreamingKCenterAggregationStrategy


class KCenterMemoryAggregationPlugin(MemoryAggParamBindingMixin, MemoryAggregationPlugin):
    """Memory aggregation plugin for `KCENTER` aggregation."""

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
                "KCENTER aggregation requires resolved n_clusters in strategy selection."
            )
        if runtime_context.training_contract.strip().upper() != "STREAMING":
            raise ValueError(
                "KCENTER aggregation is supported only in STREAMING mode "
                "(streaming coverage-coreset contract)."
            )
        params = self._mem_agg_params
        return StreamingKCenterAggregationStrategy(
            n_clusters=resolved_n_clusters,
            mode=params.kcenter_mode,
            chunk_coreset_size=params.kcenter_chunk_coreset_size,
            distance_threshold=params.kcenter_distance_threshold,
            enforce_reference_limit=params.streaming_enforce_cluster_budget,
        )
