from __future__ import annotations

from typing import Any

import pytest
import torch

from mhpc.core.checkpoint_state_adapter import build_slot_state_checkpoint_payload


def test_x54_checkpoint_state_adapter_preserves_stateful_and_stateless_slots() -> None:
    slots = {
        "feature_agg": {"opaque_state": torch.tensor([1.0], device="cpu")},
        "proj1": {"opaque_state": None},
        "transform": {},
        "mem_agg": {"opaque_state": {"count": 3}},
        "materialize": {"opaque_state": {"strategy": "generic"}},
        "scoring": {"aux_state": None},
    }

    payload = build_slot_state_checkpoint_payload(
        stage_names=slots.keys(),
        stage_slot_for=lambda stage_name: slots[stage_name],
        serialize_value=lambda value: value,
    )

    assert list(payload) == [
        "feature_agg",
        "proj1",
        "transform",
        "mem_agg",
        "materialize",
        "scoring",
    ]
    assert payload["feature_agg"]["opaque_state"].device.type == "cpu"
    assert payload["proj1"] == {"opaque_state": None}
    assert payload["transform"] == {}
    assert payload["mem_agg"] == {"opaque_state": {"count": 3}}
    assert payload["materialize"] == {"opaque_state": {"strategy": "generic"}}
    assert payload["scoring"] == {"aux_state": None}


def test_x54_checkpoint_state_adapter_rejects_non_mapping_stage_slots() -> None:
    with pytest.raises(TypeError, match="Stage checkpoint state slot must be"):
        build_slot_state_checkpoint_payload(
            stage_names=("feature_agg",),
            stage_slot_for=lambda _stage_name: ["not", "a", "mapping"],  # type: ignore[return-value]
            serialize_value=lambda value: value,
        )


def test_x54_checkpoint_state_adapter_rejects_non_mapping_serialized_slots() -> None:
    def _bad_serializer(_value: Any) -> list[str]:
        return ["not", "a", "mapping"]

    with pytest.raises(TypeError, match="Serialized stage checkpoint state slot"):
        build_slot_state_checkpoint_payload(
            stage_names=("feature_agg",),
            stage_slot_for=lambda _stage_name: {"opaque_state": object()},
            serialize_value=_bad_serializer,
        )
