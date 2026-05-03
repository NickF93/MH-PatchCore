"""`tail_aware_kcenter` memory aggregation plugin implementation."""

from __future__ import annotations

from ..contracts import (
    AggregationRuntimeState,
    MemAggRuntimeContext,
    MemoryAggregationPlugin,
)
from .param_binding import TailAwareKCenterParamBindingMixin
from .strategy import TailAwareKCenterAggregationStrategy


class TailAwareKCenterMemoryAggregationPlugin(
    TailAwareKCenterParamBindingMixin,
    MemoryAggregationPlugin,
):
    """Coverage-first k-center with a bounded sparse-tail supplement."""

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
                "tail_aware_kcenter aggregation requires resolved n_clusters."
            )
        if runtime_context.training_contract.strip().upper() != "STREAMING":
            raise ValueError(
                "tail_aware_kcenter aggregation is supported only in STREAMING mode."
            )
        params = self._mem_agg_params
        return TailAwareKCenterAggregationStrategy(
            n_clusters=resolved_n_clusters,
            chunk_coreset_size=params.kcenter_chunk_coreset_size,
            tail_selection_strategy=params.tail_selection_strategy,
            main_budget_fraction=params.main_budget_fraction,
            tail_probability_min=params.tail_probability_min,
            tail_probability_max=params.tail_probability_max,
            geometric_candidate_pool_size=params.geometric_candidate_pool_size,
            phase1_passes=params.phase1_passes,
            deduplication_strategy=params.deduplication_strategy,
            dedup_quantization_decimals=params.dedup_quantization_decimals,
            dedup_norm_tolerance=params.dedup_norm_tolerance,
            enforce_reference_limit=params.streaming_enforce_cluster_budget,
        )
