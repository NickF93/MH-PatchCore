"""Checkpoint orchestration engine for MH-PatchCore."""

from __future__ import annotations

import logging
import os
# Checkpoint artifacts are local, repository-owned files.
import pickle  # nosec B403
import typing as _T

import torch as _torch

from .checkpoint_state_adapter import build_slot_state_checkpoint_payload
from .locality_state_helpers import resolve_anomaly_scorer_memory_bank

LOGGER = logging.getLogger(__name__)


class CheckpointEngine:
    """Own checkpoint save/load and stage-state serialization semantics."""

    def __init__(self, model: _T.Any) -> None:
        self._model = model

    def _checkpoint_serialize_value(self, value: _T.Any) -> _T.Any:
        """Recursively move tensor values to CPU for portable checkpointing."""
        if isinstance(value, _torch.Tensor):
            return value.detach().to("cpu")
        if isinstance(value, dict):
            return {
                key: self._checkpoint_serialize_value(item)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [self._checkpoint_serialize_value(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self._checkpoint_serialize_value(item) for item in value)
        return value

    def _build_stage_state_checkpoint_payload(self) -> dict[str, dict[str, _T.Any]]:
        """Build the stage-owned checkpoint payload with one top-level state map."""
        return build_slot_state_checkpoint_payload(
            stage_names=self._model._stage_owned_state.keys(),
            stage_slot_for=self._model._stage_state_slot,
            serialize_value=self._checkpoint_serialize_value,
        )

    def _load_stage_state_checkpoint_payload(
        self,
        *,
        payload: dict[str, _T.Any],
    ) -> dict[str, dict[str, _T.Any]]:
        removed_patch_adapter_keys = sorted(
            str(key)
            for key in payload.keys()
            if str(key) == "patch_adapter" or str(key).startswith("patch_adapter.")
        )
        if removed_patch_adapter_keys:
            raise ValueError(
                "Checkpoint contains removed ReConPatch fields and cannot be loaded: "
                f"{', '.join(removed_patch_adapter_keys)}"
            )

        unsupported_top_level = sorted(
            str(key) for key in payload.keys() if str(key) != "stage_state"
        )
        if unsupported_top_level:
            raise ValueError(
                "Unsupported checkpoint format. "
                "Found unsupported top-level checkpoint keys: "
                f"{', '.join(unsupported_top_level)}"
            )

        if "stage_state" not in payload:
            raise ValueError("Checkpoint is missing required top-level key 'stage_state'.")
        stage_state_raw = payload["stage_state"]
        if not isinstance(stage_state_raw, dict):
            raise ValueError("Checkpoint key 'stage_state' must be a mapping.")

        expected_stages = set(self._model._stage_owned_state.keys())
        unknown_stages = sorted(
            str(stage_name)
            for stage_name in stage_state_raw.keys()
            if str(stage_name) not in expected_stages
        )
        if unknown_stages:
            raise ValueError(
                "Checkpoint contains unknown stage_state entries: "
                f"{', '.join(unknown_stages)}"
            )

        loaded_stage_state: dict[str, dict[str, _T.Any]] = {}
        for stage_name in self._model._stage_owned_state.keys():
            stage_payload = stage_state_raw.get(stage_name, {})
            if stage_payload is None:
                stage_payload = {}
            if not isinstance(stage_payload, dict):
                raise ValueError(
                    f"Checkpoint stage_state['{stage_name}'] must be a mapping."
                )
            loaded_stage_state[stage_name] = _T.cast(
                dict[str, _T.Any],
                self._checkpoint_serialize_value(stage_payload),
            )
        return loaded_stage_state

    def _require_loaded_fit_state(
        self,
        *,
        stage_name: str,
    ) -> object | None:
        state = self._model._get_stage_owned_state(
            stage_name=stage_name,
            key="opaque_state",
            default=None,
        )
        if state is None:
            raise RuntimeError(
                "Checkpoint is missing required fitted state for stage "
                f"'{stage_name}'."
            )
        return state

    def save_to_path(self, save_path: str, prepend: str = "") -> None:
        """Persist anomaly-scorer state and stage-owned checkpoint payload."""
        LOGGER.info("Saving PatchCore data.")
        os.makedirs(save_path, exist_ok=True)
        self._model.anomaly_scorer.save(
            save_path,
            save_features_separately=True,
            prepend=prepend,
        )
        patchcore_params = {
            "stage_state": self._build_stage_state_checkpoint_payload(),
        }
        with open(os.path.join(save_path, prepend + "patchcore_params.pkl"), "wb") as save_file:
            pickle.dump(patchcore_params, save_file, pickle.HIGHEST_PROTOCOL)

    def load_from_path(self, load_path: str, prepend: str = "") -> None:
        """Load anomaly-scorer state and restore validated stage-owned payloads."""
        LOGGER.info("Loading PatchCore.")
        params_path = os.path.join(load_path, prepend + "patchcore_params.pkl")
        if not os.path.exists(params_path):
            raise FileNotFoundError(f"PatchCore params file not found: {params_path}")
        with open(params_path, "rb") as load_file:
            # Local checkpoint files are trusted in this repository workflow.
            patchcore_params = pickle.load(load_file)  # nosec B301

        if not isinstance(patchcore_params, dict):
            raise ValueError("Checkpoint payload must be a mapping.")

        loaded_stage_state = self._load_stage_state_checkpoint_payload(
            payload=patchcore_params,
        )
        for stage_name, stage_slot in loaded_stage_state.items():
            self._model._stage_owned_state[stage_name] = dict(stage_slot)

        if bool(self._model._requires_feature_agg_fit_state()):
            self._require_loaded_fit_state(stage_name="feature_agg")

        proj1_state = self._model._get_stage_owned_state(
            stage_name="proj1",
            key="opaque_state",
            default=None,
        )
        if bool(self._model._uses_proj1_state()):
            proj1_state = self._require_loaded_fit_state(stage_name="proj1")
        self._model._proj1_plugin.state_load(state=proj1_state)

        transform_state = self._model._get_stage_owned_state(
            stage_name="transform",
            key="opaque_state",
            default=None,
        )
        if bool(self._model._uses_transform_state()):
            transform_state = self._require_loaded_fit_state(stage_name="transform")
        self._model._transform_plugin.state_load(state=transform_state)

        proj2_state = self._model._get_stage_owned_state(
            stage_name="proj2",
            key="opaque_state",
            default=None,
        )
        if bool(self._model._uses_proj2_state()):
            proj2_state = self._require_loaded_fit_state(stage_name="proj2")
        self._model._proj2_plugin.state_load(state=proj2_state)

        scoring_aux_state = self._model._get_stage_owned_state(
            stage_name="scoring",
            key="aux_state",
            default=None,
        )
        self._model._scoring_plugin.aux_state_validate_loaded(
            state=scoring_aux_state,
        )

        self._model.anomaly_scorer.load(load_path, prepend)
        loaded_detection_features = resolve_anomaly_scorer_memory_bank(
            anomaly_scorer=self._model.anomaly_scorer,
            stage="checkpoint load",
        )
        setattr(
            self._model.anomaly_scorer,
            "detection_features",
            loaded_detection_features,
        )
