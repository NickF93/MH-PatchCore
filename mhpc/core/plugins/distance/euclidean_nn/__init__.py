"""`euclidean_nn` distance plugin package."""

__all__ = ["EuclideanNNDistancePlugin"]


def __getattr__(name: str):
    if name == "EuclideanNNDistancePlugin":
        from .plugin import EuclideanNNDistancePlugin

        return EuclideanNNDistancePlugin
    raise AttributeError(name)
