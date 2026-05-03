"""Built-in greedy materialization plugin."""

__all__ = ["GreedyMaterializationPlugin"]


def __getattr__(name: str) -> type[object]:
    if name == "GreedyMaterializationPlugin":
        from .plugin import GreedyMaterializationPlugin

        return GreedyMaterializationPlugin
    raise AttributeError(name)
