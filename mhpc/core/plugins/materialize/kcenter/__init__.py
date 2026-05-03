"""Built-in kcenter materialization plugin."""

__all__ = ["KCenterMaterializationPlugin"]


def __getattr__(name: str) -> type[object]:
    if name == "KCenterMaterializationPlugin":
        from .plugin import KCenterMaterializationPlugin

        return KCenterMaterializationPlugin
    raise AttributeError(name)
