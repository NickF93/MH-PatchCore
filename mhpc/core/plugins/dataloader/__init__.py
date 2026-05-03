"""Built-in dataloader plugin exports."""

from pathlib import Path

from ..slot_exports import build_slot_export_map, resolve_slot_export

_EXPORT_MAP = build_slot_export_map(
    slot_package=__name__,
    slot_dir=Path(__file__).resolve().parent,
)

__all__ = sorted(_EXPORT_MAP)


def __getattr__(name: str) -> type[object]:
    return resolve_slot_export(
        name,
        export_map=_EXPORT_MAP,
    )
