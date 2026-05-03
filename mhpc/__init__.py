"""
Top-level mhpc package.

Keep imports light to avoid pulling heavy dependencies (e.g. timm) during
module discovery.
"""

__all__ = ["core", "util", "eval"]


def __getattr__(name: str) -> object:
    if name in __all__:
        import importlib

        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
