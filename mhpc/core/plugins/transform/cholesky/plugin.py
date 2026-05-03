"""Cholesky transform plugin implementation."""

from __future__ import annotations

from collections.abc import Mapping
import numpy as np
import torch
from typing import Any, cast

from mhpc.core.plugins.locality_context_contract import LocalityContext
from ..contracts import (
    TransformPlugin,
    TransformRegularizationSettings,
    TransformTrainContext,
)
from .param_binding import CholeskyTransformParamBindingMixin
from .math import (
    CovarianceRegularizationSettings,
    apply_mahalanobis_transform,
    compute_stable_cholesky,
    regularize_covariance,
    validate_covariance_regularization_settings,
)
from .streaming_covariance import StreamingCovariance


def _as_regularization_settings(
    settings: TransformRegularizationSettings,
) -> CovarianceRegularizationSettings:
    return validate_covariance_regularization_settings(
        CovarianceRegularizationSettings(
            enabled=bool(settings.enabled),
            method=str(settings.method).strip().upper(),
            shrinkage=settings.shrinkage,
            eigen_floor_ratio=float(settings.eigen_floor_ratio),
            min_jitter=float(settings.min_jitter),
            max_jitter=float(settings.max_jitter),
            jitter_multiplier=float(settings.jitter_multiplier),
        )
    )


def _storage_dtype(storage_precision: str) -> torch.dtype:
    if storage_precision == "float32":
        return torch.float32
    if storage_precision == "float64":
        return torch.float64
    raise ValueError(
        f"Unsupported storage_precision {storage_precision!r}; expected float32 or float64."
    )


class CholeskyTransformPlugin(CholeskyTransformParamBindingMixin, TransformPlugin):
    """Transform plugin with Mahalanobis/Cholesky math parity implementation."""

    supports_streaming: bool = True
    requires_full_dataset: bool = False
    requires_fit_state: bool = True
    requires_locality_context: bool = False
    preserves_locality: bool = True

    def __init__(self) -> None:
        self._train_context: TransformTrainContext | None = None
        self._cov_regularization_settings: CovarianceRegularizationSettings | None = None
        self._storage_precision = "float64"
        self._offline_batches: list[np.ndarray] = []
        self._streaming_covariance: StreamingCovariance | None = None
        self._mean: torch.Tensor | None = None
        self._cholesky_factor: torch.Tensor | None = None

    def forward_embed_transform(
        self,
        *,
        features: torch.Tensor,
        forward_modules: torch.nn.ModuleDict,
        locality_context: LocalityContext | None = None,
    ) -> torch.Tensor:
        _ = forward_modules, locality_context
        return features

    def train_start(
        self,
        *,
        context: TransformTrainContext,
    ) -> None:
        if context.training_contract not in {"OFFLINE", "STREAMING"}:
            raise ValueError(
                "Transform training contract must be OFFLINE or STREAMING; "
                f"got {context.training_contract!r}."
            )
        if context.feature_dim <= 0:
            raise ValueError("Transform feature_dim must be a positive integer.")
        self._train_context = context
        self._cov_regularization_settings = _as_regularization_settings(
            context.regularization
        )
        self._offline_batches = []
        self._streaming_covariance = None
        self._mean = None
        self._cholesky_factor = None
        if context.training_contract == "STREAMING":
            self._streaming_covariance = StreamingCovariance(
                num_features=int(context.feature_dim),
                dtype=torch.float64,
                device=torch.device("cpu"),
            )

    def train_update(
        self,
        *,
        batch: np.ndarray,
        locality_context: LocalityContext | None = None,
        update_context: object | None = None,
    ) -> None:
        _ = locality_context, update_context
        context = self._train_context
        if context is None:
            raise RuntimeError("Transform train_update called before train_start.")
        features = np.asarray(batch, dtype=np.float64)
        if features.ndim != 2:
            raise ValueError(
                "Transform train_update expects 2D features; "
                f"got shape={tuple(features.shape)}."
            )
        if int(features.shape[1]) != int(context.feature_dim):
            raise ValueError(
                "Transform train_update feature dimension mismatch: "
                f"expected {int(context.feature_dim)}, got {int(features.shape[1])}."
            )
        if context.training_contract == "OFFLINE":
            self._offline_batches.append(features.copy())
            return
        estimator = self._streaming_covariance
        if estimator is None:
            raise RuntimeError("Streaming covariance estimator is not initialized.")
        estimator.update(torch.from_numpy(features).to("cpu", dtype=torch.float64))

    def train_finalize(self) -> None:
        context = self._train_context
        settings = self._cov_regularization_settings
        if context is None or settings is None:
            raise RuntimeError("Transform train_finalize called before train_start.")

        feature_matrix: torch.Tensor | None = None
        n_samples: int
        if context.training_contract == "OFFLINE":
            if not self._offline_batches:
                raise RuntimeError(
                    "Transform OFFLINE train_finalize called with no fit batches."
                )
            offline_features = np.concatenate(self._offline_batches, axis=0)
            features_tensor = torch.from_numpy(offline_features).to(
                "cpu",
                dtype=torch.float64,
            )
            mean = torch.mean(features_tensor, dim=0)
            covariance = torch.cov(features_tensor.T)
            n_samples = int(features_tensor.shape[0])
            feature_matrix = features_tensor
        else:
            estimator = self._streaming_covariance
            if estimator is None:
                raise RuntimeError("Streaming covariance estimator is not initialized.")
            mean = estimator.mean
            covariance = estimator.covariance()
            n_samples = int(estimator.count.item())

        result = regularize_covariance(
            covariance=covariance,
            n_samples=n_samples,
            settings=settings,
            context=f"{context.training_contract.lower()}_transform_fit",
            feature_matrix=feature_matrix,
        )
        factor = compute_stable_cholesky(
            covariance=result.covariance,
            context=f"{context.training_contract.lower()}_transform_fit",
            settings=settings,
        )
        self._mean = mean.detach().to("cpu", dtype=torch.float64)
        self._cholesky_factor = factor.detach().to("cpu", dtype=torch.float64)
        self._offline_batches = []
        self._streaming_covariance = None

    def infer_transform(
        self,
        *,
        features: np.ndarray,
        stage: str,
        batch_idx: int | None = None,
        locality_context: LocalityContext | None = None,
    ) -> np.ndarray:
        _ = locality_context
        if self._mean is None or self._cholesky_factor is None:
            raise RuntimeError(
                "Transform-stage state is not initialized. "
                "Call fit() before infer()."
            )
        return apply_mahalanobis_transform(
            features=np.asarray(features, dtype=np.float64),
            mean=self._mean,
            cholesky_factor=self._cholesky_factor,
            stage=stage,
            batch_idx=batch_idx,
        )

    def state_export(self) -> object | None:
        if self._mean is None or self._cholesky_factor is None:
            return None
        storage_dtype = _storage_dtype(self._storage_precision)
        return {
            "mean": self._mean.detach().to("cpu", dtype=storage_dtype),
            "cholesky_factor": self._cholesky_factor.detach().to(
                "cpu",
                dtype=storage_dtype,
            ),
        }

    def state_load(
        self,
        *,
        state: object | None,
    ) -> None:
        self._train_context = None
        self._cov_regularization_settings = None
        self._offline_batches = []
        self._streaming_covariance = None
        self._mean = None
        self._cholesky_factor = None
        if state is None:
            return
        if not isinstance(state, Mapping):
            raise ValueError("Transform checkpoint state must be a mapping or null.")
        payload = cast(Mapping[str, Any], state)
        mean = payload.get("mean")
        cholesky_factor = payload.get("cholesky_factor")
        if not isinstance(mean, torch.Tensor):
            raise ValueError(
                "Transform checkpoint state key 'mean' must be a torch.Tensor."
            )
        if not isinstance(cholesky_factor, torch.Tensor):
            raise ValueError(
                "Transform checkpoint state key 'cholesky_factor' must be a torch.Tensor."
            )
        mean_t = mean.detach().to("cpu")
        cholesky_t = cholesky_factor.detach().to("cpu")
        if not torch.is_floating_point(mean_t):
            raise ValueError("Transform checkpoint mean tensor must be floating point.")
        if not torch.is_floating_point(cholesky_t):
            raise ValueError(
                "Transform checkpoint cholesky_factor tensor must be floating point."
            )
        if mean_t.ndim != 1:
            raise ValueError("Transform checkpoint mean tensor must be rank-1.")
        if cholesky_t.ndim != 2 or cholesky_t.shape[0] != cholesky_t.shape[1]:
            raise ValueError(
                "Transform checkpoint cholesky_factor tensor must be a square matrix."
            )
        if int(cholesky_t.shape[0]) != int(mean_t.shape[0]):
            raise ValueError(
                "Transform checkpoint state dimension mismatch between mean and "
                "cholesky_factor."
            )
        self._mean = mean_t.to(dtype=torch.float64)
        self._cholesky_factor = cholesky_t.to(dtype=torch.float64)
