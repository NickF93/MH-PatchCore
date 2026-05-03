import numpy as np
import torch

from mhpc.core.plugins.transform.cholesky.math import (
    CovarianceRegularizationSettings,
    apply_mahalanobis_transform,
    compute_stable_cholesky,
    regularize_covariance,
)


def test_cholesky_transform_matches_mahalanobis_quadratic_form() -> None:
    covariance = torch.tensor(
        [[2.0, 0.3], [0.3, 0.8]],
        dtype=torch.float64,
    )
    mean = torch.tensor([1.0, -2.0], dtype=torch.float64)
    settings = CovarianceRegularizationSettings(
        enabled=False,
        method="JITTER_ONLY",
        shrinkage="auto",
        eigen_floor_ratio=0.0,
        min_jitter=1.0e-12,
        max_jitter=1.0e-3,
        jitter_multiplier=10.0,
    )

    factor = compute_stable_cholesky(covariance, "unit-test", settings)
    features = np.array([[2.0, -1.0], [0.5, -2.5]], dtype=np.float64)
    transformed = apply_mahalanobis_transform(features, mean, factor, "unit-test")

    inv_cov = torch.linalg.inv(covariance).numpy()
    deltas = features - mean.numpy()[None, :]
    expected_sq = np.einsum("ni,ij,nj->n", deltas, inv_cov, deltas)
    actual_sq = np.sum(transformed**2, axis=1)
    np.testing.assert_allclose(actual_sq, expected_sq, rtol=1.0e-10, atol=1.0e-10)


def test_regularization_makes_singular_covariance_factorable() -> None:
    covariance = torch.ones((3, 3), dtype=torch.float64)
    settings = CovarianceRegularizationSettings(
        enabled=True,
        method="FIXED",
        shrinkage=0.25,
        eigen_floor_ratio=1.0e-8,
        min_jitter=1.0e-12,
        max_jitter=1.0,
        jitter_multiplier=10.0,
    )

    regularized = regularize_covariance(
        covariance,
        n_samples=4,
        settings=settings,
        context="unit-test",
    )
    factor = compute_stable_cholesky(regularized.covariance, "unit-test", settings)
    assert factor.shape == covariance.shape
    assert torch.isfinite(factor).all()
