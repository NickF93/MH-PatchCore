"""Built-in `kcenter` memory aggregation plugin."""

__all__ = ["KCenterMemoryAggregationPlugin"]


def __getattr__(name: str) -> type[object]:
    if name == "KCenterMemoryAggregationPlugin":
        from .plugin import KCenterMemoryAggregationPlugin

        return KCenterMemoryAggregationPlugin
    raise AttributeError(name)
