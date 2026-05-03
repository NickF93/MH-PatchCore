"""Built-in reference-max scoring plugin."""

__all__ = ["ReferenceMaxScoringPlugin"]


def __getattr__(name: str) -> type[object]:
    if name == "ReferenceMaxScoringPlugin":
        from .plugin import ReferenceMaxScoringPlugin

        return ReferenceMaxScoringPlugin
    raise AttributeError(name)
