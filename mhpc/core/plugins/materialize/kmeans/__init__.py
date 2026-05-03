"""Built-in kmeans materialization plugin."""

__all__ = ["KMeansMaterializationPlugin"]


def __getattr__(name: str) -> type[object]:
    if name == "KMeansMaterializationPlugin":
        from .plugin import KMeansMaterializationPlugin

        return KMeansMaterializationPlugin
    raise AttributeError(name)
