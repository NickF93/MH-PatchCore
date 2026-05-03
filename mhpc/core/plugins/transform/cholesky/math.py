"""Covariance regularization, stable Cholesky, and Mahalanobis utilities."""

from __future__ import annotations

from dataclasses import dataclass
import logging

import numpy as np
import torch
from sklearn.covariance import LedoitWolf  # type: ignore[import-untyped]

LOGGER = logging.getLogger(__name__)

_VALID_REGULARIZATION_METHODS = frozenset(
    {"OAS", "LEDOIT_WOLF", "FIXED", "JITTER_ONLY"}
)


@dataclass(frozen=True)
class CovarianceRegularizationSettings:
    """Validated covariance regularization settings."""

    enabled: bool
    method: str
    shrinkage: str | float
    eigen_floor_ratio: float
    min_jitter: float
    max_jitter: float
    jitter_multiplier: float


@dataclass(frozen=True)
class CovarianceRegularizationResult:
    """Result bundle for covariance regularization."""

    covariance: torch.Tensor
    shrinkage: float


def validate_covariance_regularization_settings(
    settings: CovarianceRegularizationSettings,
) -> CovarianceRegularizationSettings:
    """Validate covariance regularization settings with deterministic errors."""
    if not isinstance(settings.enabled, bool):
        raise ValueError("covariance_regularization_enabled must be a boolean")

    method = settings.method
    if method not in _VALID_REGULARIZATION_METHODS:
        raise ValueError(
            "covariance_regularization_method must be one of "
            f"{sorted(_VALID_REGULARIZATION_METHODS)}"
        )

    if settings.eigen_floor_ratio < 0.0:
        raise ValueError("covariance_eigen_floor_ratio must be >= 0")
    if settings.min_jitter <= 0.0:
        raise ValueError("covariance_min_jitter must be > 0")
    if settings.max_jitter < settings.min_jitter:
        raise ValueError("covariance_max_jitter must be >= covariance_min_jitter")
    if settings.jitter_multiplier <= 1.0:
        raise ValueError("covariance_jitter_multiplier must be > 1")

    shrinkage = settings.shrinkage
    if method == "FIXED":
        if shrinkage == "auto":
            raise ValueError(
                "covariance_regularization_shrinkage must be a float for FIXED method"
            )
        if isinstance(shrinkage, bool) or not isinstance(shrinkage, (float, int)):
            raise ValueError(
                "covariance_regularization_shrinkage must be float or 'auto'"
            )
        shrinkage_value = float(shrinkage)
        if not np.isfinite(shrinkage_value):
            raise ValueError("covariance_regularization_shrinkage must be finite")
        if not 0.0 <= shrinkage_value <= 1.0:
            raise ValueError(
                "covariance_regularization_shrinkage must be in [0, 1]"
            )
        return CovarianceRegularizationSettings(
            enabled=settings.enabled,
            method=method,
            shrinkage=shrinkage_value,
            eigen_floor_ratio=settings.eigen_floor_ratio,
            min_jitter=settings.min_jitter,
            max_jitter=settings.max_jitter,
            jitter_multiplier=settings.jitter_multiplier,
        )

    if shrinkage != "auto":
        raise ValueError(
            "covariance_regularization_shrinkage must be 'auto' for non-FIXED methods"
        )

    return settings


def validate_finite_feature_array(
    features: np.ndarray,
    stage: str,
    batch_idx: int | None = None,
) -> None:
    """Fail fast when non-finite feature values are detected."""
    finite_mask = np.isfinite(features)
    if np.all(finite_mask):
        return

    non_finite_count = int(features.size - int(finite_mask.sum()))
    batch_suffix = f", batch={batch_idx}" if batch_idx is not None else ""
    raise ValueError(
        f"Non-finite feature values detected during {stage}{batch_suffix}. "
        f"non_finite_count={non_finite_count}"
    )


def compute_stable_cholesky(
    covariance: torch.Tensor,
    context: str,
    settings: CovarianceRegularizationSettings,
) -> torch.Tensor:
    """Compute a stable Cholesky factor using a bounded adaptive jitter schedule."""
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError(
            f"Covariance must be a 2D square matrix, got shape={tuple(covariance.shape)}"
        )

    # Perform numerically sensitive factorization on CPU float64.
    cov = covariance.detach().to("cpu", dtype=torch.float64)
    cov = 0.5 * (cov + cov.T)

    finite_mask = torch.isfinite(cov)
    if not bool(finite_mask.all().item()):
        non_finite_count = int((~finite_mask).sum().item())
        raise ValueError(
            f"Covariance contains non-finite values during {context}. "
            f"non_finite_count={non_finite_count}"
        )

    diagonal = torch.diagonal(cov)
    min_diag = float(diagonal.min().item())
    max_diag = float(diagonal.max().item())
    eye = torch.eye(cov.shape[0], device="cpu", dtype=cov.dtype)

    jitter = 0.0
    attempts = 0
    while True:
        attempts += 1
        candidate = cov if jitter == 0.0 else cov + eye * jitter
        success = False
        try:
            if hasattr(torch.linalg, "cholesky_ex"):
                factor, info = torch.linalg.cholesky_ex(
                    candidate,
                    check_errors=False,
                )
                success = int(info.max().item()) == 0 and bool(
                    torch.isfinite(factor).all().item()
                )
            else:
                factor = torch.linalg.cholesky(candidate)
                success = bool(torch.isfinite(factor).all().item())
        except RuntimeError:
            success = False

        if success:
            if jitter > 0.0:
                LOGGER.warning(
                    "Applied jitter=%.1e to covariance during %s for stable Cholesky "
                    "(attempts=%d).",
                    jitter,
                    context,
                    attempts,
                )
            else:
                LOGGER.info(
                    "Computed stable Cholesky without jitter during %s (attempts=%d).",
                    context,
                    attempts,
                )
            return factor

        if jitter == 0.0:
            jitter = settings.min_jitter
        else:
            jitter *= settings.jitter_multiplier

        if jitter > settings.max_jitter:
            break

    min_eigenvalue = float("nan")
    try:
        min_eigenvalue = float(torch.linalg.eigvalsh(cov).min().item())
    except RuntimeError:
        LOGGER.exception("Failed to compute covariance eigenvalues for diagnostics.")

    raise RuntimeError(
        "Failed to compute a stable Cholesky factor. "
        f"context={context}, min_diag={min_diag:.6e}, max_diag={max_diag:.6e}, "
        f"min_eigenvalue={min_eigenvalue:.6e}"
    )


def apply_mahalanobis_transform(
    features: np.ndarray,
    mean: torch.Tensor,
    cholesky_factor: torch.Tensor,
    stage: str,
    batch_idx: int | None = None,
) -> np.ndarray:
    """Apply z = L^-1 (x - mean) in float64 on CPU for solver stability."""
    validate_finite_feature_array(
        features=features,
        stage=f"{stage}_input",
        batch_idx=batch_idx,
    )

    l_cpu = cholesky_factor.detach().to("cpu", dtype=torch.float64)
    mean_cpu = mean.detach().to("cpu", dtype=torch.float64)
    features_tensor = torch.from_numpy(features).to("cpu", dtype=torch.float64)
    transformed = torch.linalg.solve_triangular(
        l_cpu,
        (features_tensor - mean_cpu).T,
        upper=False,
    ).T
    transformed_np = transformed.numpy()
    validate_finite_feature_array(
        features=transformed_np,
        stage=f"{stage}_output",
        batch_idx=batch_idx,
    )
    return transformed_np


def compute_oas_shrinkage(covariance: torch.Tensor, n_samples: int) -> float:
    """Compute OAS shrinkage coefficient from sample covariance."""
    if n_samples <= 1:
        return 1.0

    dim = int(covariance.shape[0])
    if dim <= 1:
        return 1.0

    trace_cov = float(torch.trace(covariance).item())
    trace_cov_sq = trace_cov * trace_cov
    trace_cov2 = float(torch.trace(covariance @ covariance).item())

    denominator = (n_samples + 1.0 - 2.0 / dim) * (
        trace_cov2 - (trace_cov_sq / dim)
    )
    if denominator <= 0.0:
        return 1.0

    numerator = (1.0 - 2.0 / dim) * trace_cov2 + trace_cov_sq
    shrinkage = numerator / denominator
    return float(np.clip(shrinkage, 0.0, 1.0))


def compute_rblw_shrinkage(covariance: torch.Tensor, n_samples: int) -> float:
    """Compute Rao-Blackwell Ledoit-Wolf shrinkage coefficient."""
    if n_samples <= 2:
        return 1.0

    dim = int(covariance.shape[0])
    if dim <= 1:
        return 1.0

    trace_cov = float(torch.trace(covariance).item())
    trace_cov_sq = trace_cov * trace_cov
    trace_cov2 = float(torch.trace(covariance @ covariance).item())
    denominator = (n_samples + 2.0) * (trace_cov2 - (trace_cov_sq / dim))
    if denominator <= 0.0:
        return 1.0

    numerator = ((n_samples - 2.0) / n_samples) * trace_cov2 + trace_cov_sq
    shrinkage = numerator / denominator
    return float(np.clip(shrinkage, 0.0, 1.0))


def regularize_covariance(
    covariance: torch.Tensor,
    n_samples: int,
    settings: CovarianceRegularizationSettings,
    context: str,
    feature_matrix: torch.Tensor | None = None,
) -> CovarianceRegularizationResult:
    """Regularize covariance matrix with configured shrinkage/eigen-floor policy."""
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError(
            f"Covariance must be a 2D square matrix, got shape={tuple(covariance.shape)}"
        )
    if n_samples <= 0:
        raise ValueError(f"n_samples must be > 0 during {context}. got={n_samples}")

    cov = covariance.detach().to("cpu", dtype=torch.float64)
    cov = 0.5 * (cov + cov.T)

    finite_mask = torch.isfinite(cov)
    if not bool(finite_mask.all().item()):
        non_finite_count = int((~finite_mask).sum().item())
        raise ValueError(
            f"Covariance contains non-finite values during {context}. "
            f"non_finite_count={non_finite_count}"
        )

    if not settings.enabled:
        LOGGER.info(
            "Covariance regularization disabled for %s. method=%s",
            context,
            settings.method,
        )
        return CovarianceRegularizationResult(covariance=cov, shrinkage=0.0)

    shrinkage = _resolve_shrinkage(
        covariance=cov,
        n_samples=n_samples,
        settings=settings,
        context=context,
        feature_matrix=feature_matrix,
    )

    if settings.method != "JITTER_ONLY":
        dim = int(cov.shape[0])
        mu = float(torch.trace(cov).item()) / max(dim, 1)
        identity = torch.eye(dim, device="cpu", dtype=cov.dtype)
        cov = (1.0 - shrinkage) * cov + shrinkage * mu * identity

        if settings.eigen_floor_ratio > 0.0:
            eigvals, eigvecs = torch.linalg.eigh(cov)
            floor_value = settings.eigen_floor_ratio * max(abs(mu), 1.0e-12)
            eigvals = torch.clamp(eigvals, min=floor_value)
            cov = eigvecs @ torch.diag(eigvals) @ eigvecs.T
            cov = 0.5 * (cov + cov.T)
            min_eigenvalue = float(eigvals.min().item())
        else:
            min_eigenvalue = float(torch.linalg.eigvalsh(cov).min().item())
    else:
        min_eigenvalue = float(torch.linalg.eigvalsh(cov).min().item())

    LOGGER.info(
        "Covariance regularization for %s: enabled=%s method=%s shrinkage=%.6f "
        "n_samples=%d min_eigenvalue=%.6e eigen_floor_ratio=%.3e",
        context,
        settings.enabled,
        settings.method,
        shrinkage,
        n_samples,
        min_eigenvalue,
        settings.eigen_floor_ratio,
    )
    return CovarianceRegularizationResult(covariance=cov, shrinkage=shrinkage)


def _resolve_shrinkage(
    covariance: torch.Tensor,
    n_samples: int,
    settings: CovarianceRegularizationSettings,
    context: str,
    feature_matrix: torch.Tensor | None = None,
) -> float:
    if settings.method == "FIXED":
        return float(settings.shrinkage)
    if settings.method == "OAS":
        return compute_oas_shrinkage(covariance, n_samples)
    if settings.method == "LEDOIT_WOLF":
        if feature_matrix is not None:
            lw = LedoitWolf(assume_centered=False)
            lw.fit(feature_matrix.detach().cpu().numpy())
            return float(np.clip(lw.shrinkage_, 0.0, 1.0))

        shrinkage = compute_rblw_shrinkage(covariance, n_samples)
        LOGGER.warning(
            "Using RBLW approximation for LEDOIT_WOLF shrinkage during %s "
            "because full feature matrix is unavailable.",
            context,
        )
        return shrinkage

    return 0.0
