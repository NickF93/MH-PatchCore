"""Built-in `kmeans` memory aggregation plugin."""

__all__ = ["KMeansMemoryAggregationPlugin"]


def __getattr__(name: str) -> type[object]:
    if name == "KMeansMemoryAggregationPlugin":
        from .plugin import KMeansMemoryAggregationPlugin

        return KMeansMemoryAggregationPlugin
    raise AttributeError(name)
