"""Plugin-local parameter binding for the Cholesky transform plugin."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal, cast

from ..contracts import (
    TransformBindContextLike,
    TransformRegularizationSettings,
    TransformTrainContext,
)

_ALLOWED_KEYS = frozenset({"covariance_regularization", "storage_precision"})
_ALLOWED_COV_REG_KEYS = frozenset(
    {
        "enabled",
        "method",
        "shrinkage",
        "eigen_floor_ratio",
        "min_jitter",
        "max_jitter",
        "jitter_multiplier",
    }
)
_ALLOWED_COV_REG_METHODS = frozenset({"OAS", "LEDOIT_WOLF", "FIXED", "JITTER_ONLY"})
_ALLOWED_STORAGE_PRECISIONS = frozenset({"float64", "float32"})
_FIXED_METHOD = "FIXED"


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


class CholeskyTransformParamBindingMixin:
    """Plugin-local bind-time parser for Cholesky transform parameters."""

    _bound_params: dict[str, Any]
    _bound_bind_context: TransformBindContextLike
    _regularization_settings: TransformRegularizationSettings
    _storage_precision: str

    def bind_params(
        self,
        *,
        params: Mapping[str, Any],
        bind_context: TransformBindContextLike,
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
        unknown_keys = sorted(
            str(key) for key in set(self._bound_params.keys()) - _ALLOWED_KEYS
        )
        if unknown_keys:
            raise ValueError(
                "pipeline.slots.transform.params contains unsupported keys: "
                f"{', '.join(unknown_keys)}"
            )
        raw_cov_reg = self._bound_params.get("covariance_regularization", {})
        if not isinstance(raw_cov_reg, Mapping):
            raise ValueError("covariance_regularization must be a mapping.")
        unknown_cov_keys = sorted(
            str(key) for key in set(raw_cov_reg.keys()) - _ALLOWED_COV_REG_KEYS
        )
        if unknown_cov_keys:
            raise ValueError(
                "covariance_regularization contains unsupported keys: "
                f"{', '.join(unknown_cov_keys)}"
            )

        enabled = bool(raw_cov_reg.get("enabled", True))
        method = str(raw_cov_reg.get("method", "OAS")).strip().upper()
        if method not in _ALLOWED_COV_REG_METHODS:
            raise ValueError(
                "covariance_regularization.method must be one of: "
                f"{', '.join(sorted(_ALLOWED_COV_REG_METHODS))}."
            )
        shrinkage = raw_cov_reg.get("shrinkage", "auto")
        if isinstance(shrinkage, str):
            if shrinkage != "auto":
                raise ValueError(
                    "covariance_regularization.shrinkage must be 'auto' or float [0, 1]."
                )
        elif isinstance(shrinkage, (float, int)) and not isinstance(shrinkage, bool):
            shrinkage_float = float(shrinkage)
            if not 0.0 <= shrinkage_float <= 1.0:
                raise ValueError(
                    "covariance_regularization.shrinkage must be 'auto' or float [0, 1]."
                )
            shrinkage = shrinkage_float
        else:
            raise ValueError(
                "covariance_regularization.shrinkage must be 'auto' or float [0, 1]."
            )

        if method == _FIXED_METHOD and shrinkage == "auto":
            raise ValueError(
                "covariance_regularization.shrinkage must be float for FIXED method."
            )
        if method != _FIXED_METHOD and shrinkage != "auto":
            raise ValueError(
                "covariance_regularization.shrinkage must be 'auto' for non-FIXED methods."
            )

        eigen_floor_ratio = float(raw_cov_reg.get("eigen_floor_ratio", 1.0e-8))
        if eigen_floor_ratio < 0.0:
            raise ValueError("covariance_regularization.eigen_floor_ratio must be >= 0.")
        min_jitter = float(raw_cov_reg.get("min_jitter", 1.0e-12))
        if min_jitter <= 0.0:
            raise ValueError("covariance_regularization.min_jitter must be > 0.")
        max_jitter = float(raw_cov_reg.get("max_jitter", 1.0))
        if max_jitter < min_jitter:
            raise ValueError(
                "covariance_regularization.max_jitter must be >= min_jitter."
            )
        jitter_multiplier = float(raw_cov_reg.get("jitter_multiplier", 10.0))
        if jitter_multiplier <= 1.0:
            raise ValueError(
                "covariance_regularization.jitter_multiplier must be > 1."
            )

        self._regularization_settings = TransformRegularizationSettings(
            enabled=enabled,
            method=method,
            shrinkage=shrinkage,
            eigen_floor_ratio=eigen_floor_ratio,
            min_jitter=min_jitter,
            max_jitter=max_jitter,
            jitter_multiplier=jitter_multiplier,
        )
        storage_precision = str(
            self._bound_params.get("storage_precision", "float64")
        ).strip().lower()
        if storage_precision not in _ALLOWED_STORAGE_PRECISIONS:
            raise ValueError(
                "storage_precision must be one of: float32, float64."
            )
        self._storage_precision = storage_precision

    def resolve_train_context(
        self,
        *,
        training_contract: Literal["OFFLINE", "STREAMING"],
        feature_dim: int,
    ) -> TransformTrainContext:
        normalized_contract = _normalize_training_contract(training_contract)
        if feature_dim <= 0:
            raise ValueError("Transform feature_dim must be a positive integer.")
        return TransformTrainContext(
            training_contract=cast(Literal["OFFLINE", "STREAMING"], normalized_contract),
            feature_dim=int(feature_dim),
            regularization=self._regularization_settings,
        )
