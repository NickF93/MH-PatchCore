"""Plugin-local parameter binding/parsing for `reference_max` scoring."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..contracts import ScoringBindContextLike, ScoringRuntimeControls

_ALLOWED_KEYS = frozenset({"paper_reweight_num_nn", "patch_scoring"})
_ALLOWED_PATCH_SCORING_KEYS = frozenset({"mode", "pni"})
_ALLOWED_PNI_KEYS = frozenset(
    {
        "prototype_source",
        "train_view_policy",
        "neighborhood_kernel_size",
        "neighborhood_use_relative",
        "prior_mix_gamma",
        "position_laplace_alpha",
        "faithful_prior_threshold",
        "faithful_distance_scale",
        "topk_k",
        "topk_temperature",
        "topp_p",
        "topp_max_k",
        "soft_distance_alpha",
        "eps",
        "classifier_hidden_dim",
        "classifier_epochs",
        "classifier_updates_per_batch",
        "classifier_learning_rate",
        "classifier_weight_decay",
        "assignment_chunk_size",
        "prototype_chunk_size",
        "logits_chunk_size",
        "dist_coreset_size",
        "dist_coreset_minibatch_size",
        "dist_coreset_random_state",
    }
)
_GLOBAL_TOPK_SOFT_ALLOWED_PNI_KEYS = frozenset({"topk_k", "topk_temperature"})
_GLOBAL_TOPP_MEAN_ALLOWED_PNI_KEYS = frozenset({"topp_p", "topp_max_k"})
_SUPPORTED_PATCH_SCORING_MODES = frozenset(
    {
        "GLOBAL_ONLY",
        "GLOBAL_TOPK_SOFT",
        "GLOBAL_TOPP_MEAN",
        "PNI_FAITHFUL_GATE",
        "PNI_SOFT_FUSION",
    }
)


class ScoringParamBindingMixin:
    """Scoring slot mixin with plugin-local scoring control parsing."""

    _bound_params: dict[str, Any]
    _bound_bind_context: ScoringBindContextLike
    _scoring_controls: ScoringRuntimeControls

    def bind_params(
        self,
        *,
        params: Mapping[str, Any],
        bind_context: ScoringBindContextLike,
    ) -> None:
        if not isinstance(params, Mapping):
            raise TypeError(
                "params must be a mapping for plugin bind_params: "
                f"type={type(params).__name__}"
            )
        training_contract = getattr(bind_context, "training_contract", None)
        if not isinstance(training_contract, str):
            raise TypeError(
                "bind_context.training_contract must be a string: "
                f"type={type(training_contract).__name__}"
            )
        normalized_contract = training_contract.strip().upper()
        if normalized_contract not in {"OFFLINE", "STREAMING"}:
            raise ValueError(
                "bind_context.training_contract must be one of "
                "{'OFFLINE', 'STREAMING'}: "
                f"value={training_contract!r}"
            )
        seed = getattr(bind_context, "seed", None)
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError(
                "bind_context.seed must be an integer: "
                f"type={type(seed).__name__}"
            )
        self._bound_params = dict(params)
        self._bound_bind_context = bind_context
        unknown_keys = sorted(str(key) for key in set(self._bound_params.keys()) - _ALLOWED_KEYS)
        if unknown_keys:
            raise ValueError(
                "pipeline.slots.scoring.params contains unsupported keys: "
                f"{', '.join(unknown_keys)}"
            )

        raw_reweight = self._bound_params.get("paper_reweight_num_nn", 9)
        if isinstance(raw_reweight, bool) or not isinstance(raw_reweight, int) or raw_reweight <= 0:
            raise ValueError("paper_reweight_num_nn must be an integer > 0.")
        paper_reweight_num_nn = int(raw_reweight)

        raw_patch_scoring = self._bound_params.get("patch_scoring", {})
        if not isinstance(raw_patch_scoring, Mapping):
            raise ValueError("patch_scoring must be a mapping.")
        unknown_patch_scoring_keys = sorted(
            str(key) for key in set(raw_patch_scoring.keys()) - _ALLOWED_PATCH_SCORING_KEYS
        )
        if unknown_patch_scoring_keys:
            raise ValueError(
                "patch_scoring contains unsupported keys: "
                f"{', '.join(unknown_patch_scoring_keys)}"
            )
        patch_scoring_mode = str(raw_patch_scoring.get("mode", "GLOBAL_ONLY")).strip().upper()
        if not patch_scoring_mode:
            raise ValueError("patch_scoring.mode must be a non-empty string token.")
        if patch_scoring_mode not in _SUPPORTED_PATCH_SCORING_MODES:
            raise ValueError(
                "patch_scoring.mode must be one of "
                "{'GLOBAL_ONLY', 'GLOBAL_TOPK_SOFT', 'GLOBAL_TOPP_MEAN', "
                "'PNI_FAITHFUL_GATE', 'PNI_SOFT_FUSION'}."
            )

        raw_pni = raw_patch_scoring.get("pni", {})
        if not isinstance(raw_pni, Mapping):
            raise ValueError("patch_scoring.pni must be a mapping.")
        unknown_pni_keys = sorted(str(key) for key in set(raw_pni.keys()) - _ALLOWED_PNI_KEYS)
        if unknown_pni_keys:
            raise ValueError(
                "patch_scoring.pni contains unsupported keys: "
                f"{', '.join(unknown_pni_keys)}"
            )
        if patch_scoring_mode == "GLOBAL_TOPK_SOFT":
            disallowed_pni_keys = sorted(
                str(key)
                for key in set(raw_pni.keys()) - _GLOBAL_TOPK_SOFT_ALLOWED_PNI_KEYS
            )
            if disallowed_pni_keys:
                raise ValueError(
                    "patch_scoring.pni contains unsupported keys for "
                    "patch_scoring.mode=GLOBAL_TOPK_SOFT: "
                    f"{', '.join(disallowed_pni_keys)}"
                )
        elif patch_scoring_mode == "GLOBAL_TOPP_MEAN":
            disallowed_pni_keys = sorted(
                str(key)
                for key in set(raw_pni.keys()) - _GLOBAL_TOPP_MEAN_ALLOWED_PNI_KEYS
            )
            if disallowed_pni_keys:
                raise ValueError(
                    "patch_scoring.pni contains unsupported keys for "
                    "patch_scoring.mode=GLOBAL_TOPP_MEAN: "
                    f"{', '.join(disallowed_pni_keys)}"
                )
        else:
            disallowed_topk_keys = sorted(
                str(key) for key in set(raw_pni.keys()) & _GLOBAL_TOPK_SOFT_ALLOWED_PNI_KEYS
            )
            if disallowed_topk_keys:
                raise ValueError(
                    "patch_scoring.pni contains unsupported keys for "
                    f"patch_scoring.mode={patch_scoring_mode}: "
                    f"{', '.join(disallowed_topk_keys)}"
                )
            disallowed_topp_keys = sorted(
                str(key) for key in set(raw_pni.keys()) & _GLOBAL_TOPP_MEAN_ALLOWED_PNI_KEYS
            )
            if disallowed_topp_keys:
                raise ValueError(
                    "patch_scoring.pni contains unsupported keys for "
                    f"patch_scoring.mode={patch_scoring_mode}: "
                    f"{', '.join(disallowed_topp_keys)}"
                )

        pni_prototype_source = str(raw_pni.get("prototype_source", "MEMORY_BANK")).strip().upper()
        if not pni_prototype_source:
            raise ValueError("patch_scoring.pni.prototype_source must be a non-empty token.")
        pni_train_view_policy = str(raw_pni.get("train_view_policy", "CURRENT_POLICY")).strip().upper()
        if not pni_train_view_policy:
            raise ValueError("patch_scoring.pni.train_view_policy must be a non-empty token.")

        pni_neighborhood_kernel_size = int(raw_pni.get("neighborhood_kernel_size", 3))
        if pni_neighborhood_kernel_size <= 0 or pni_neighborhood_kernel_size % 2 == 0:
            raise ValueError("patch_scoring.pni.neighborhood_kernel_size must be a positive odd integer.")
        pni_neighborhood_use_relative = raw_pni.get("neighborhood_use_relative", True)
        if not isinstance(pni_neighborhood_use_relative, bool):
            raise ValueError("patch_scoring.pni.neighborhood_use_relative must be a boolean.")
        pni_prior_mix_gamma = float(raw_pni.get("prior_mix_gamma", 0.5))
        if not 0.0 <= pni_prior_mix_gamma <= 1.0:
            raise ValueError("patch_scoring.pni.prior_mix_gamma must be in [0, 1].")
        pni_position_laplace_alpha = float(raw_pni.get("position_laplace_alpha", 1.0))
        if pni_position_laplace_alpha <= 0.0:
            raise ValueError("patch_scoring.pni.position_laplace_alpha must be > 0.")
        pni_faithful_prior_threshold = float(raw_pni.get("faithful_prior_threshold", 0.01))
        if not 0.0 <= pni_faithful_prior_threshold < 1.0:
            raise ValueError("patch_scoring.pni.faithful_prior_threshold must be in [0, 1).")
        pni_faithful_distance_scale = float(raw_pni.get("faithful_distance_scale", 1.0))
        if pni_faithful_distance_scale <= 0.0:
            raise ValueError("patch_scoring.pni.faithful_distance_scale must be > 0.")
        pni_assignment_chunk_size = int(raw_pni.get("assignment_chunk_size", 8192))
        if pni_assignment_chunk_size <= 0:
            raise ValueError("patch_scoring.pni.assignment_chunk_size must be > 0.")
        pni_prototype_chunk_size = int(raw_pni.get("prototype_chunk_size", 4096))
        if pni_prototype_chunk_size <= 0:
            raise ValueError("patch_scoring.pni.prototype_chunk_size must be > 0.")
        pni_topk_k = int(raw_pni.get("topk_k", 5))
        if pni_topk_k < 1:
            raise ValueError("patch_scoring.pni.topk_k must be >= 1.")
        pni_topk_temperature = float(raw_pni.get("topk_temperature", 1.0))
        if pni_topk_temperature <= 0.0:
            raise ValueError("patch_scoring.pni.topk_temperature must be > 0.")
        pni_topp_p = float(raw_pni.get("topp_p", 0.90))
        if not 0.0 < pni_topp_p <= 1.0:
            raise ValueError("patch_scoring.pni.topp_p must be in (0, 1].")
        pni_topp_max_k = int(raw_pni.get("topp_max_k", 32))
        if pni_topp_max_k < 1:
            raise ValueError("patch_scoring.pni.topp_max_k must be >= 1.")

        # Keep parity checks for currently accepted numeric knobs, even when not
        # used by the active scoring algorithm.
        if not 0.0 <= float(raw_pni.get("soft_distance_alpha", 0.8)) <= 1.0:
            raise ValueError("patch_scoring.pni.soft_distance_alpha must be in [0, 1].")
        if float(raw_pni.get("eps", 1.0e-12)) <= 0.0:
            raise ValueError("patch_scoring.pni.eps must be > 0.")
        if int(raw_pni.get("classifier_hidden_dim", 256)) <= 0:
            raise ValueError("patch_scoring.pni.classifier_hidden_dim must be > 0.")
        if int(raw_pni.get("classifier_epochs", 1)) <= 0:
            raise ValueError("patch_scoring.pni.classifier_epochs must be > 0.")
        if int(raw_pni.get("classifier_updates_per_batch", 1)) <= 0:
            raise ValueError("patch_scoring.pni.classifier_updates_per_batch must be > 0.")
        if float(raw_pni.get("classifier_learning_rate", 1.0e-3)) <= 0.0:
            raise ValueError("patch_scoring.pni.classifier_learning_rate must be > 0.")
        if float(raw_pni.get("classifier_weight_decay", 1.0e-6)) < 0.0:
            raise ValueError("patch_scoring.pni.classifier_weight_decay must be >= 0.")
        if int(raw_pni.get("logits_chunk_size", 4096)) <= 0:
            raise ValueError("patch_scoring.pni.logits_chunk_size must be > 0.")
        if int(raw_pni.get("dist_coreset_size", 256)) <= 0:
            raise ValueError("patch_scoring.pni.dist_coreset_size must be > 0.")
        if int(raw_pni.get("dist_coreset_minibatch_size", 256)) <= 0:
            raise ValueError("patch_scoring.pni.dist_coreset_minibatch_size must be > 0.")
        if int(raw_pni.get("dist_coreset_random_state", 42)) < 0:
            raise ValueError("patch_scoring.pni.dist_coreset_random_state must be >= 0.")

        self._scoring_controls = ScoringRuntimeControls(
            patch_scoring_mode=patch_scoring_mode,
            paper_reweight_num_nn=paper_reweight_num_nn,
            pni_prototype_source=pni_prototype_source,
            pni_train_view_policy=pni_train_view_policy,
            pni_neighborhood_kernel_size=pni_neighborhood_kernel_size,
            pni_neighborhood_use_relative=pni_neighborhood_use_relative,
            pni_prior_mix_gamma=pni_prior_mix_gamma,
            pni_position_laplace_alpha=pni_position_laplace_alpha,
            pni_faithful_prior_threshold=pni_faithful_prior_threshold,
            pni_faithful_distance_scale=pni_faithful_distance_scale,
            pni_assignment_chunk_size=pni_assignment_chunk_size,
            pni_prototype_chunk_size=pni_prototype_chunk_size,
            pni_topk_k=pni_topk_k,
            pni_topk_temperature=pni_topk_temperature,
            pni_topp_p=pni_topp_p,
            pni_topp_max_k=pni_topp_max_k,
        )

    def resolve_scoring_controls(self) -> ScoringRuntimeControls:
        return self._scoring_controls
