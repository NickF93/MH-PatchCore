"""Built-in `tail_aware_kcenter` memory aggregation plugin."""

__all__ = ["TailAwareKCenterMemoryAggregationPlugin"]


def __getattr__(name: str) -> type[object]:
    if name == "TailAwareKCenterMemoryAggregationPlugin":
        from .plugin import TailAwareKCenterMemoryAggregationPlugin

        return TailAwareKCenterMemoryAggregationPlugin
    raise AttributeError(name)
