"""Shared progress-bar helpers.

This module centralizes tqdm defaults to keep rendering behavior consistent
across fit/predict/evaluation loops without changing loop semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import sys
from typing import Any, TextIO

import tqdm.auto as _tqdm  # type: ignore[import-untyped]


DEFAULT_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
DATASETS_PROGRESS_DESC = "Datasets"
INFERENCE_PROGRESS_DESC = "Inferring..."


def metrics_progress_desc(dataset_name: str) -> str:
    """Return the canonical metrics progress-bar description."""
    return f"Metrics[{dataset_name}]"


def artifacts_progress_desc(dataset_name: str) -> str:
    """Return the canonical artifacts progress-bar description."""
    return f"Artifacts[{dataset_name}]"


def calibration_progress_desc(dataset_name: str) -> str:
    """Return the canonical calibration progress-bar description."""
    return f"Calibration[{dataset_name}]"


@dataclass
class ProgressRenderSettings:
    """Mutable project-wide rendering defaults for progress bars."""

    enabled: bool = True
    leave: bool = True
    dynamic_ncols: bool = True
    min_interval: float = 0.1


_SETTINGS = ProgressRenderSettings()


class TqdmAwareLoggingHandler(logging.Handler):
    """Logging handler that writes through tqdm when progress bars are active."""

    def __init__(self, stream: TextIO | None = None) -> None:
        super().__init__()
        self._stream = stream if stream is not None else sys.stderr

    def emit(self, record: logging.LogRecord) -> None:
        """Emit one record while preserving active tqdm bars."""
        try:
            message = self.format(record)
            # tqdm.write is safe both with and without active bars.
            _tqdm.tqdm.write(message, file=self._stream)
        except Exception:
            self.handleError(record)


def configure_progress_rendering(
    *,
    enabled: bool,
    leave: bool,
    dynamic_ncols: bool,
    min_interval: float,
) -> None:
    """Update process-wide progress-bar defaults."""
    _SETTINGS.enabled = enabled
    _SETTINGS.leave = leave
    _SETTINGS.dynamic_ncols = dynamic_ncols
    _SETTINGS.min_interval = min_interval


def configure_root_logger(
    *,
    level: str | int = logging.INFO,
    fmt: str = DEFAULT_LOG_FORMAT,
    datefmt: str = DEFAULT_DATE_FORMAT,
) -> None:
    """Configure root logger with a tqdm-aware stream handler."""
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(level)

    handler = TqdmAwareLoggingHandler()
    handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
    root_logger.addHandler(handler)


def create_progress_bar(*args: Any, **kwargs: Any) -> Any:
    """Create a tqdm progress bar with project-wide defaults.

    Defaults are controlled by :func:`configure_progress_rendering` and may be
    overridden per call through explicit keyword arguments.
    """
    kwargs.setdefault("disable", not _SETTINGS.enabled)
    kwargs.setdefault("leave", _SETTINGS.leave)
    kwargs.setdefault("dynamic_ncols", _SETTINGS.dynamic_ncols)
    kwargs.setdefault("mininterval", _SETTINGS.min_interval)
    return _tqdm.tqdm(*args, **kwargs)


def make_progress_postfix(
    *,
    batch: int | str | None = None,
    batch_size: int | str | None = None,
    images: int | str | None = None,
    phase: str | None = None,
    saved: int | str | None = None,
    total: int | str | None = None,
) -> dict[str, int | str]:
    """Create a standardized postfix payload for tqdm bars."""
    payload: dict[str, int | str] = {}
    if batch is not None:
        payload["batch"] = batch
    if batch_size is not None:
        payload["batch_size"] = batch_size
    if images is not None:
        payload["images"] = images
    if phase is not None:
        payload["phase"] = phase
    if saved is not None:
        payload["saved"] = saved
    if total is not None:
        payload["total"] = total
    return payload
