"""Canonical slot-parameter ownership primitives (MS-108).

This module intentionally exposes only canonical slot metadata and explicitly
allowed non-slot globals. Legacy slot-path ownership maps and migration
placement discriminators were removed in MS-108.
"""

from __future__ import annotations

from dataclasses import dataclass

from mhpc.core.pipeline_stage_contract import canonical_slot_stages

_CANONICAL_SLOTS: tuple[str, ...] = canonical_slot_stages()

_ALLOWED_MODEL_ROOT_KEYS: frozenset[str] = frozenset()
_ALLOWED_MODEL_PATCH_CORE_KEYS: frozenset[str] = frozenset()


@dataclass(frozen=True)
class NonSlotGlobalParameter:
    """One authoritative non-slot global path outside pipeline slot params."""

    path: str


NON_SLOT_GLOBAL_PARAMETERS: tuple[NonSlotGlobalParameter, ...] = (
    NonSlotGlobalParameter("experiment.name"),
    NonSlotGlobalParameter("experiment.seed"),
    NonSlotGlobalParameter("paths.output_root"),
    NonSlotGlobalParameter("runtime.device"),
    NonSlotGlobalParameter("pipeline.training.contract"),
    NonSlotGlobalParameter("pipeline.training.fit_epochs"),
    NonSlotGlobalParameter("evaluation"),
    NonSlotGlobalParameter("artifacts"),
    NonSlotGlobalParameter("teacher_export"),
    NonSlotGlobalParameter("render"),
)

_NON_SLOT_GLOBAL_PATHS: frozenset[str] = frozenset(
    entry.path for entry in NON_SLOT_GLOBAL_PARAMETERS
)


def canonical_slots() -> tuple[str, ...]:
    """Return canonical runtime slot order."""

    return _CANONICAL_SLOTS


def allowed_model_root_keys() -> frozenset[str]:
    """Return canonical allowed keys under top-level `model`."""

    return _ALLOWED_MODEL_ROOT_KEYS


def allowed_model_patch_core_keys() -> frozenset[str]:
    """Return canonical non-slot global keys under `model.patch_core`."""

    return _ALLOWED_MODEL_PATCH_CORE_KEYS


def is_non_slot_global_parameter(path: str) -> bool:
    """Return true if path is intentionally outside pipeline slot params."""

    if path in _NON_SLOT_GLOBAL_PATHS:
        return True
    return path.startswith("pipeline.training.fit_epochs.")
