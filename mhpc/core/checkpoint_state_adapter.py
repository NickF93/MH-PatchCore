"""Generic checkpoint-state adapters for slot-owned opaque payloads."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any, cast


def build_slot_state_checkpoint_payload(
    *,
    stage_names: Iterable[str],
    stage_slot_for: Callable[[str], Mapping[str, Any]],
    serialize_value: Callable[[Any], Any],
) -> dict[str, dict[str, Any]]:
    """Build a checkpoint payload from stage-owned state with no plugin branching."""
    payload: dict[str, dict[str, Any]] = {}
    for stage_name in stage_names:
        stage_slot = stage_slot_for(stage_name)
        if not isinstance(stage_slot, Mapping):
            raise TypeError(
                "Stage checkpoint state slot must be a mapping: "
                f"stage='{stage_name}' type='{type(stage_slot).__name__}'"
            )
        serialized_slot = serialize_value(dict(stage_slot))
        if not isinstance(serialized_slot, dict):
            raise TypeError(
                "Serialized stage checkpoint state slot must be a mapping: "
                f"stage='{stage_name}' type='{type(serialized_slot).__name__}'"
            )
        payload[stage_name] = cast(dict[str, Any], serialized_slot)
    return payload


__all__ = ["build_slot_state_checkpoint_payload"]
