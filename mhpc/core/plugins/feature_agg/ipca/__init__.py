"""Built-in IPCA feature-aggregation plugin."""

__all__ = ["IPCAFeatureAggregatorPlugin"]


def __getattr__(name: str) -> type[object]:
    if name == "IPCAFeatureAggregatorPlugin":
        from .plugin import IPCAFeatureAggregatorPlugin

        return IPCAFeatureAggregatorPlugin
    raise AttributeError(name)
