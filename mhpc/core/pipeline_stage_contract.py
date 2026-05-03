"""Host-owned canonical stage-order contracts for capability-driven orchestration."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

ExecutionMode = Literal["train", "inference"]
StageModeCapabilityMap = Mapping[str, Mapping[str, bool]]
_MODE_CAPABILITY_KEYS: Mapping[ExecutionMode, str] = {
    "train": "supports_train",
    "inference": "supports_inference",
}

CANONICAL_SLOT_STAGES: tuple[str, ...] = (
    "dataloader",
    "backbone",
    "patch_align",
    "preprocess",
    "feature_agg",
    "proj1",
    "transform",
    "proj2",
    "mem_agg",
    "materialize",
    "distance",
    "scoring",
)

# Kept as contract aliases: both execution paths traverse the same canonical order.
TRAIN_PIPELINE_STAGE_ORDER: tuple[str, ...] = CANONICAL_SLOT_STAGES
INFERENCE_PIPELINE_STAGE_ORDER: tuple[str, ...] = CANONICAL_SLOT_STAGES


def _assert_unique_stage_sequence(stage_names: tuple[str, ...], *, surface_name: str) -> None:
    if len(stage_names) != len(set(stage_names)):
        raise ValueError(
            f"{surface_name} must contain unique stage names; got duplicates in {stage_names!r}"
        )


def _validate_stage_contracts() -> None:
    _assert_unique_stage_sequence(
        CANONICAL_SLOT_STAGES,
        surface_name="CANONICAL_SLOT_STAGES",
    )
    _assert_unique_stage_sequence(
        TRAIN_PIPELINE_STAGE_ORDER,
        surface_name="TRAIN_PIPELINE_STAGE_ORDER",
    )
    _assert_unique_stage_sequence(
        INFERENCE_PIPELINE_STAGE_ORDER,
        surface_name="INFERENCE_PIPELINE_STAGE_ORDER",
    )

    canonical = set(CANONICAL_SLOT_STAGES)
    train_set = set(TRAIN_PIPELINE_STAGE_ORDER)
    inference_set = set(INFERENCE_PIPELINE_STAGE_ORDER)

    unknown_train = sorted(train_set - canonical)
    if unknown_train:
        raise ValueError(
            "TRAIN_PIPELINE_STAGE_ORDER declares unknown canonical stages: "
            f"{', '.join(unknown_train)}"
        )

    unknown_inference = sorted(inference_set - canonical)
    if unknown_inference:
        raise ValueError(
            "INFERENCE_PIPELINE_STAGE_ORDER declares unknown canonical stages: "
            f"{', '.join(unknown_inference)}"
        )


def canonical_slot_stages() -> tuple[str, ...]:
    """Return the canonical fixed slot order for pluginized runtime surfaces."""

    return CANONICAL_SLOT_STAGES


def train_pipeline_stage_order() -> tuple[str, ...]:
    """Return the canonical fixed stage order for TRAIN execution."""

    return TRAIN_PIPELINE_STAGE_ORDER


def inference_pipeline_stage_order() -> tuple[str, ...]:
    """Return the canonical fixed stage order for INFERENCE execution."""

    return INFERENCE_PIPELINE_STAGE_ORDER


def stage_execution_role(stage_name: str) -> str:
    """Return capability-driven role token for one canonical stage."""

    if stage_name not in CANONICAL_SLOT_STAGES:
        raise KeyError(f"Unknown canonical stage '{stage_name}'.")
    return "capability_driven"


def is_stage_allowed_in_mode(
    stage_name: str,
    *,
    execution_mode: ExecutionMode,
    stage_mode_capabilities: StageModeCapabilityMap | None = None,
) -> bool:
    """Return whether a stage is executable in the given mode from plugin capabilities."""

    if stage_name not in CANONICAL_SLOT_STAGES:
        raise KeyError(f"Unknown canonical stage '{stage_name}'.")
    if stage_mode_capabilities is None:
        return True
    stage_caps = stage_mode_capabilities.get(stage_name)
    if stage_caps is None:
        raise KeyError(
            "Missing stage mode capabilities for canonical stage: "
            f"stage='{stage_name}'."
        )
    capability_key = _MODE_CAPABILITY_KEYS[execution_mode]
    capability_value = stage_caps.get(capability_key)
    if not isinstance(capability_value, bool):
        raise ValueError(
            "Stage mode capability must be boolean: "
            f"stage='{stage_name}' capability='{capability_key}' "
            f"type='{type(capability_value).__name__}'"
        )
    return capability_value


def assert_stage_allowed_in_mode(
    stage_name: str,
    *,
    execution_mode: ExecutionMode,
    stage_mode_capabilities: StageModeCapabilityMap | None = None,
) -> None:
    """Fail fast when a stage is executed outside plugin-mode capabilities."""

    if is_stage_allowed_in_mode(
        stage_name,
        execution_mode=execution_mode,
        stage_mode_capabilities=stage_mode_capabilities,
    ):
        return
    raise ValueError(
        "Stage mode capability violation: "
        f"stage='{stage_name}' mode='{execution_mode}'."
    )


_validate_stage_contracts()
