"""Built-in no-op transform plugin."""

__all__ = ["NoneTransformPlugin"]


def __getattr__(name: str) -> type[object]:
    if name == "NoneTransformPlugin":
        from .plugin import NoneTransformPlugin

        return NoneTransformPlugin
    raise AttributeError(name)
