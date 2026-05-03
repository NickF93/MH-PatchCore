"""Built-in `greedy` memory aggregation plugin."""

__all__ = ["GreedyMemoryAggregationPlugin"]


def __getattr__(name: str) -> type[object]:
    if name == "GreedyMemoryAggregationPlugin":
        from .plugin import GreedyMemoryAggregationPlugin

        return GreedyMemoryAggregationPlugin
    raise AttributeError(name)
