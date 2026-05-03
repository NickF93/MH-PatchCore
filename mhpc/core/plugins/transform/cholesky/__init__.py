"""Built-in Cholesky transform plugin."""

__all__ = ["CholeskyTransformPlugin"]


def __getattr__(name: str) -> type[object]:
    if name == "CholeskyTransformPlugin":
        from .plugin import CholeskyTransformPlugin

        return CholeskyTransformPlugin
    raise AttributeError(name)
