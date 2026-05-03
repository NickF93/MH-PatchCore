"""Built-in paper-eq7 scoring plugin."""

__all__ = ["PaperEq7ScoringPlugin"]


def __getattr__(name: str) -> type[object]:
    if name == "PaperEq7ScoringPlugin":
        from .plugin import PaperEq7ScoringPlugin

        return PaperEq7ScoringPlugin
    raise AttributeError(name)
