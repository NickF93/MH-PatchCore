"""Plugin-local parameter binding/parsing for `feature_agg:none`."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ..contracts import FeatureAggBindContextLike, ReductionSelection

_ALLOWED_KEYS = frozenset({"target_embed_dimension"})


@dataclass(frozen=True)
class FeatureAggParams:
    target_embed_dimension: int


def _normalize_training_contract(value: object) -> str:
    raw_value = value.name if hasattr(value, "name") else value
    if not isinstance(raw_value, str):
        raise TypeError(
            "training_contract must be a string token: "
            f"type={type(raw_value).__name__}"
        )
    normalized = raw_value.strip().upper()
    if normalized not in {"OFFLINE", "STREAMING"}:
        raise ValueError(
            "training_contract must be one of {'OFFLINE', 'STREAMING'}: "
            f"value={raw_value!r}"
        )
    return normalized


def _ensure_allowed_keys(*, params: Mapping[str, Any], context: str) -> None:
    unknown_keys = sorted(str(key) for key in set(params.keys()) - _ALLOWED_KEYS)
    if unknown_keys:
        raise ValueError(f"{context} contains unsupported keys: {', '.join(unknown_keys)}")


def _require_positive_int(*, value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer.")
    return int(value)


def _parse_feature_agg_params(params: Mapping[str, Any]) -> FeatureAggParams:
    _ensure_allowed_keys(params=params, context="pipeline.slots.feature_agg.params")
    return FeatureAggParams(
        target_embed_dimension=_require_positive_int(
            value=params.get("target_embed_dimension"),
            field_name="target_embed_dimension",
        )
    )


class FeatureAggParamBindingMixin:
    """Feature-aggregation param parsing localized to `none`."""

    _bound_params: dict[str, Any]
    _bound_bind_context: FeatureAggBindContextLike
    _feature_agg_params: FeatureAggParams

    def bind_params(
        self,
        *,
        params: Mapping[str, Any],
        bind_context: FeatureAggBindContextLike,
    ) -> None:
        if not isinstance(params, Mapping):
            raise TypeError(
                "params must be a mapping for plugin bind_params: "
                f"type={type(params).__name__}"
            )
        _normalize_training_contract(getattr(bind_context, "training_contract", None))
        seed = getattr(bind_context, "seed", None)
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError(
                "bind_context.seed must be an integer: "
                f"type={type(seed).__name__}"
            )
        self._bound_params = dict(params)
        self._bound_bind_context = bind_context
        self._feature_agg_params = _parse_feature_agg_params(self._bound_params)

    def resolve_target_embed_dimension(self) -> int:
        return int(self._feature_agg_params.target_embed_dimension)

    def resolve_reduction_selection(
        self,
        *,
        training_contract: str,
    ) -> ReductionSelection:
        normalized_contract = _normalize_training_contract(training_contract)
        algorithm = "STREAMING" if normalized_contract == "STREAMING" else "VANILLA"
        return ReductionSelection(
            algorithm=algorithm,
            pca_variance_ratio=1.0,
        )
