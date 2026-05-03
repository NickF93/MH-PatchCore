"""`greedy` memory aggregation plugin implementation."""

from __future__ import annotations

from ..contracts import (
    AggregationRuntimeState,
    MemAggRuntimeContext,
    MemoryAggregationPlugin,
)
from .param_binding import MemoryAggParamBindingMixin
from .strategy import GreedyCoresetAggregationStrategy


class GreedyMemoryAggregationPlugin(MemoryAggParamBindingMixin, MemoryAggregationPlugin):
    """Memory aggregation plugin for `GREEDY` coreset selection."""

    supports_streaming: bool = False
    requires_full_dataset: bool = True
    requires_locality_context: bool = False
    preserves_locality: bool = False

    def create_runtime_state(
        self,
        *,
        runtime_context: MemAggRuntimeContext,
    ) -> AggregationRuntimeState:
        params = self._mem_agg_params
        return GreedyCoresetAggregationStrategy(
            percentage=params.coreset_percentage,
            device=runtime_context.device,
        )
