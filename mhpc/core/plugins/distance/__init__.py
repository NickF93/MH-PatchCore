"""Built-in distance plugin exports."""

from functools import lru_cache
from pathlib import Path

from ..slot_exports import build_slot_export_map, resolve_slot_export

# Keep slot-export map lazy to avoid import cycles during contract bootstrap.
__all__: list[str] = []


@lru_cache(maxsize=1)
def _export_map() -> dict[str, tuple[str, str]]:
    return build_slot_export_map(
        slot_package=__name__,
        slot_dir=Path(__file__).resolve().parent,
    )


def __getattr__(name: str) -> type[object]:
    return resolve_slot_export(
        name,
        export_map=_export_map(),
    )


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_export_map().keys()))
