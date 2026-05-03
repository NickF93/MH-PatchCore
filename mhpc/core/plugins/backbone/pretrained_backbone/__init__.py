"""Built-in pretrained backbone plugin."""

__all__ = ["PretrainedBackbonePlugin"]


def __getattr__(name: str) -> type[object]:
    if name == "PretrainedBackbonePlugin":
        from .plugin import PretrainedBackbonePlugin

        return PretrainedBackbonePlugin
    raise AttributeError(name)
