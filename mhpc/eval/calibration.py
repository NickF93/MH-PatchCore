"""Score calibration utilities for image-level and pixel-level anomaly outputs."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import math

import numpy as np

LOGGER = logging.getLogger(__name__)

_ALLOWED_CALIBRATION_MODES = frozenset({"none", "zscore", "ecdf"})


@dataclass
class _OnlineMoments:
    """Streaming moments state used by z-score calibration."""

    count: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def update(self, values: np.ndarray) -> None:
        flat_values = np.asarray(values, dtype=np.float64).reshape(-1)
        if flat_values.size == 0:
            raise ValueError("Calibration update batch must be non-empty.")

        batch_count = int(flat_values.size)
        batch_mean = float(flat_values.mean())
        batch_var = float(flat_values.var())
        batch_m2 = batch_var * float(batch_count)

        if self.count == 0:
            self.count = batch_count
            self.mean = batch_mean
            self.m2 = batch_m2
            return

        total_count = self.count + batch_count
        delta = batch_mean - self.mean
        self.mean += delta * (batch_count / total_count)
        self.m2 += (
            batch_m2
            + (delta * delta) * ((self.count * batch_count) / total_count)
        )
        self.count = total_count

    @property
    def std(self) -> float:
        if self.count <= 0:
            return 0.0
        variance = max(self.m2 / float(self.count), 0.0)
        return math.sqrt(variance)


@dataclass(frozen=True)
class CalibrationConfig:
    """Calibration settings used by evaluation pipeline."""

    mode: str
    eps: float
    apply_to_image: bool
    apply_to_pixel: bool


class ScoreCalibrator:
    """Calibrate anomaly score scales with deterministic transforms."""

    def __init__(self, config: CalibrationConfig) -> None:
        if config.mode not in _ALLOWED_CALIBRATION_MODES:
            raise ValueError(
                f"Unsupported calibration mode: {config.mode}. "
                f"Allowed: {sorted(_ALLOWED_CALIBRATION_MODES)}"
            )
        if config.eps <= 0.0:
            raise ValueError("calibration.eps must be > 0")

        self._config = config
        self._image_mean: float | None = None
        self._image_std: float | None = None
        self._pixel_mean: float | None = None
        self._pixel_std: float | None = None
        self._image_sorted: np.ndarray | None = None
        self._pixel_sorted: np.ndarray | None = None
        self._image_moments: _OnlineMoments | None = None
        self._pixel_moments: _OnlineMoments | None = None
        self._has_online_updates = False
        self._fitted = False

        if config.mode == "zscore":
            self._reset_zscore_state()

    @property
    def mode(self) -> str:
        return self._config.mode

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def fit(self, image_scores: np.ndarray, pixel_maps: np.ndarray) -> "ScoreCalibrator":
        """Fit calibration parameters on train-split model outputs."""
        image_arr = _require_image_scores(image_scores)
        pixel_arr = _require_pixel_maps(pixel_maps)
        if image_arr.shape[0] != pixel_arr.shape[0]:
            raise ValueError("image_scores and pixel_maps must share batch size")

        mode = self._config.mode
        if mode == "none":
            self._fitted = True
            return self

        if mode == "zscore":
            self._reset_zscore_state()
            self.update(image_arr, pixel_arr)
            self.finalize_fit()
            return self

        # mode == "ecdf"
        if self._config.apply_to_image:
            self._image_sorted = np.sort(image_arr.astype(np.float64, copy=False))
        if self._config.apply_to_pixel:
            self._pixel_sorted = np.sort(pixel_arr.reshape(-1).astype(np.float64, copy=False))
        self._fitted = True
        return self

    def update(self, image_scores: np.ndarray, pixel_maps: np.ndarray) -> "ScoreCalibrator":
        """Update z-score calibration statistics from one streamed batch."""
        if self._config.mode != "zscore":
            raise RuntimeError("update() is supported only when calibration mode is 'zscore'.")
        if self._fitted:
            raise RuntimeError("Cannot update ScoreCalibrator after it has been fitted.")

        image_arr = _require_image_scores(image_scores)
        pixel_arr = _require_pixel_maps(pixel_maps)
        if image_arr.shape[0] != pixel_arr.shape[0]:
            raise ValueError("image_scores and pixel_maps must share batch size")

        if self._config.apply_to_image:
            image_moments = self._require_moments(self._image_moments, "image moments")
            image_moments.update(image_arr)
        if self._config.apply_to_pixel:
            pixel_moments = self._require_moments(self._pixel_moments, "pixel moments")
            pixel_moments.update(pixel_arr)

        self._has_online_updates = True
        return self

    def finalize_fit(self) -> "ScoreCalibrator":
        """Finalize z-score calibration after one or more streaming updates."""
        if self._config.mode != "zscore":
            raise RuntimeError(
                "finalize_fit() is supported only when calibration mode is 'zscore'."
            )
        if self._fitted:
            raise RuntimeError("ScoreCalibrator is already fitted.")
        if not self._has_online_updates:
            raise RuntimeError("Cannot finalize zscore calibration without update() calls.")

        if self._config.apply_to_image:
            image_moments = self._require_moments(self._image_moments, "image moments")
            self._image_mean = image_moments.mean
            self._image_std = image_moments.std
        if self._config.apply_to_pixel:
            pixel_moments = self._require_moments(self._pixel_moments, "pixel moments")
            self._pixel_mean = pixel_moments.mean
            self._pixel_std = pixel_moments.std

        self._fitted = True
        return self

    def transform(
        self,
        image_scores: np.ndarray,
        pixel_maps: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply fitted calibration mapping."""
        if not self._fitted:
            raise RuntimeError("ScoreCalibrator must be fitted before transform().")

        image_arr = _require_image_scores(image_scores)
        pixel_arr = _require_pixel_maps(pixel_maps)
        if image_arr.shape[0] != pixel_arr.shape[0]:
            raise ValueError("image_scores and pixel_maps must share batch size")

        mode = self._config.mode
        if mode == "none":
            return image_arr, pixel_arr

        image_out = image_arr
        pixel_out = pixel_arr

        if mode == "zscore":
            if self._config.apply_to_image:
                image_out = _zscore_transform(
                    image_arr,
                    mean=self._require_value(self._image_mean, "image mean"),
                    std=self._require_value(self._image_std, "image std"),
                    eps=self._config.eps,
                )
            if self._config.apply_to_pixel:
                pixel_out = _zscore_transform(
                    pixel_arr,
                    mean=self._require_value(self._pixel_mean, "pixel mean"),
                    std=self._require_value(self._pixel_std, "pixel std"),
                    eps=self._config.eps,
                )
            return image_out, pixel_out

        # mode == "ecdf"
        if self._config.apply_to_image:
            image_sorted = self._require_array(self._image_sorted, "image ECDF support")
            image_out = _ecdf_transform(image_arr, image_sorted)
        if self._config.apply_to_pixel:
            pixel_sorted = self._require_array(self._pixel_sorted, "pixel ECDF support")
            pixel_out = _ecdf_transform(pixel_arr, pixel_sorted)
        return image_out, pixel_out

    @staticmethod
    def _require_value(value: float | None, name: str) -> float:
        if value is None:
            raise RuntimeError(f"Missing fitted calibration parameter: {name}.")
        return value

    @staticmethod
    def _require_moments(
        value: _OnlineMoments | None,
        name: str,
    ) -> _OnlineMoments:
        if value is None:
            raise RuntimeError(f"Missing zscore calibration state: {name}.")
        return value

    def _reset_zscore_state(self) -> None:
        self._image_mean = None
        self._image_std = None
        self._pixel_mean = None
        self._pixel_std = None
        self._image_moments = _OnlineMoments() if self._config.apply_to_image else None
        self._pixel_moments = _OnlineMoments() if self._config.apply_to_pixel else None
        self._has_online_updates = False
        self._fitted = False

    @staticmethod
    def _require_array(value: np.ndarray | None, name: str) -> np.ndarray:
        if value is None:
            raise RuntimeError(f"Missing fitted calibration parameter: {name}.")
        return value


def build_score_calibrator(config: CalibrationConfig) -> ScoreCalibrator | None:
    """Create calibrator only when mode is enabled."""
    if config.mode == "none":
        return None
    calibrator = ScoreCalibrator(config=config)
    LOGGER.info(
        "Enabled score calibration: mode=%s apply_to_image=%s apply_to_pixel=%s",
        config.mode,
        config.apply_to_image,
        config.apply_to_pixel,
    )
    if config.mode == "ecdf" and config.apply_to_pixel:
        LOGGER.warning(
            "ECDF pixel calibration stores all train pixel scores in memory; "
            "this stage is not streaming."
        )
    return calibrator


def _zscore_transform(
    values: np.ndarray,
    mean: float,
    std: float,
    eps: float,
) -> np.ndarray:
    denominator = max(std, eps)
    return (values - mean) / denominator


def _ecdf_transform(values: np.ndarray, sorted_support: np.ndarray) -> np.ndarray:
    support = sorted_support.reshape(-1)
    if support.size == 0:
        raise ValueError("ECDF support must be non-empty.")
    ranks = np.searchsorted(support, values, side="right")
    return ranks.astype(np.float64, copy=False) / float(support.size)


def _require_image_scores(image_scores: np.ndarray) -> np.ndarray:
    arr = np.asarray(image_scores, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"image_scores must be 1D, got shape={arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError("image_scores contain non-finite values")
    return arr


def _require_pixel_maps(pixel_maps: np.ndarray) -> np.ndarray:
    arr = np.asarray(pixel_maps, dtype=np.float64)
    if arr.ndim != 3:
        raise ValueError(f"pixel_maps must be 3D [N,H,W], got shape={arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError("pixel_maps contain non-finite values")
    return arr
