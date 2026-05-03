"""Resource profiling utilities for per-dataset experiment tracking.

Collects time-series samples of CPU utilization, RAM, VRAM, and optional GPU
utilization via a background polling thread. Exact wall-clock durations are
recorded for named pipeline phases.
"""

from __future__ import annotations

import atexit
import logging
import threading
import time
from dataclasses import dataclass
from typing import Any

import psutil  # type: ignore[import-untyped]
import torch

LOGGER = logging.getLogger(__name__)

RESOURCE_SAMPLE_COLUMNS: tuple[str, ...] = (
    "dataset",
    "elapsed_s",
    "ram_mb",
    "vram_mb",
    "cpu_pct",
    "gpu_util_pct",
)

PHASE_RECORD_COLUMNS: tuple[str, ...] = (
    "dataset",
    "phase",
    "duration_s",
)

_PYNVML_AVAILABLE = False
_pynvml: Any | None = None


def _shutdown_nvml() -> None:
    """Release NVML resources if initialized.

    Safe to call multiple times.
    """
    global _PYNVML_AVAILABLE

    if not _PYNVML_AVAILABLE or _pynvml is None:
        return

    try:
        _pynvml.nvmlShutdown()
    except Exception as exc:  # pragma: no cover - depends on NVML runtime state
        LOGGER.debug("NVML shutdown failed: %s", exc)
    finally:
        _PYNVML_AVAILABLE = False


try:
    import pynvml as _pynvml_module  # type: ignore[import-not-found,import-untyped]

    _pynvml_module.nvmlInit()
    _pynvml = _pynvml_module
    _PYNVML_AVAILABLE = True
    atexit.register(_shutdown_nvml)
    LOGGER.debug("pynvml initialized; GPU utilization tracking enabled.")
except Exception as exc:  # pragma: no cover - optional dependency path
    LOGGER.debug("pynvml unavailable (%s); gpu_util_pct will be NaN.", exc)


@dataclass(frozen=True)
class ResourceSample:
    """Single time-stamped resource measurement."""

    dataset: str
    elapsed_s: float
    ram_mb: float
    vram_mb: float
    cpu_pct: float
    gpu_util_pct: float


@dataclass(frozen=True)
class PhaseRecord:
    """Wall-clock duration for one named pipeline phase."""

    dataset: str
    phase: str
    duration_s: float


def phases_to_timing_dict(phases: list[PhaseRecord]) -> dict[str, float]:
    """Convert phase records to a phase -> duration mapping."""
    return {phase.phase: phase.duration_s for phase in phases}


class DatasetProfiler:
    """Collect full time-series resource data for each dataset run."""

    def __init__(
        self,
        device: torch.device,
        poll_interval_s: float = 0.5,
    ) -> None:
        if poll_interval_s <= 0.0:
            raise ValueError(
                f"poll_interval_s must be > 0, got {poll_interval_s}."
            )

        self._device = device
        self._poll_interval = float(poll_interval_s)
        self._process = psutil.Process()
        self._process.cpu_percent()

        self._nvml_handle: Any | None = None
        if _PYNVML_AVAILABLE and _pynvml is not None and device.type == "cuda":
            try:
                device_idx = (
                    int(device.index)
                    if device.index is not None
                    else int(torch.cuda.current_device())
                )
                self._nvml_handle = _pynvml.nvmlDeviceGetHandleByIndex(device_idx)
            except Exception as exc:  # pragma: no cover - GPU/NVML runtime dependent
                LOGGER.debug(
                    "Could not obtain NVML handle for device=%s: %s",
                    device,
                    exc,
                )

        self._dataset_name = ""
        self._t_dataset_start = 0.0
        self._t_phase_start = 0.0
        self._dataset_active = False

        self._samples: list[ResourceSample] = []
        self._phases: list[PhaseRecord] = []
        self._lock = threading.Lock()

        self._stop_event = threading.Event()
        self._poll_thread: threading.Thread | None = None

    def start_dataset(self, dataset_name: str) -> None:
        """Begin profiling for one dataset category."""
        if not dataset_name:
            raise ValueError("dataset_name must be a non-empty string.")

        self._stop_event.set()
        self._join_poll_thread()

        self._dataset_name = dataset_name
        self._t_dataset_start = time.perf_counter()
        self._t_phase_start = 0.0
        self._dataset_active = True

        with self._lock:
            self._samples = []
            self._phases = []

        self._process.cpu_percent()

        self._stop_event.clear()
        self._poll_thread = threading.Thread(
            target=self._poll_loop,
            name=f"profiler-poll-{dataset_name}",
            daemon=True,
        )
        self._poll_thread.start()

    def start_phase(self) -> None:
        """Record phase start timestamp for the active dataset."""
        if not self._dataset_active:
            raise RuntimeError("start_phase() called without an active dataset.")
        if self._t_phase_start != 0.0:
            raise RuntimeError(
                "start_phase() called while another phase is already active."
            )
        self._t_phase_start = time.perf_counter()

    def end_phase(self, name: str) -> None:
        """Record phase end timestamp and persist its duration."""
        if not self._dataset_active:
            raise RuntimeError("end_phase() called without an active dataset.")
        if not name:
            raise ValueError("Phase name must be a non-empty string.")
        if self._t_phase_start == 0.0:
            raise RuntimeError(
                f"end_phase('{name}') called without a preceding start_phase()."
            )

        duration_s = time.perf_counter() - self._t_phase_start
        with self._lock:
            self._phases.append(
                PhaseRecord(
                    dataset=self._dataset_name,
                    phase=name,
                    duration_s=duration_s,
                )
            )
        self._t_phase_start = 0.0

    def finish_dataset(self) -> tuple[list[ResourceSample], list[PhaseRecord]]:
        """Stop polling and finalize records for the active dataset."""
        if not self._dataset_active:
            raise RuntimeError(
                "finish_dataset() called without a preceding start_dataset()."
            )

        if self._t_phase_start != 0.0:
            raise RuntimeError(
                "finish_dataset() called while a phase is active. "
                "Call end_phase() first."
            )

        total_s = time.perf_counter() - self._t_dataset_start

        self._stop_event.set()
        self._join_poll_thread()

        with self._lock:
            self._phases.append(
                PhaseRecord(
                    dataset=self._dataset_name,
                    phase="total",
                    duration_s=total_s,
                )
            )
            samples = list(self._samples)
            phases = list(self._phases)

        self._dataset_active = False
        return samples, phases

    def _join_poll_thread(self) -> None:
        if self._poll_thread is None:
            return
        self._poll_thread.join(timeout=2.0)
        if self._poll_thread.is_alive():
            LOGGER.warning(
                "Profiler polling thread did not stop within timeout for dataset=%s.",
                self._dataset_name,
            )
        self._poll_thread = None

    def _poll_loop(self) -> None:
        while not self._stop_event.wait(timeout=self._poll_interval):
            elapsed_s = time.perf_counter() - self._t_dataset_start

            try:
                ram_mb = self._process.memory_info().rss / (1024.0**2)
            except Exception:
                ram_mb = float("nan")

            vram_mb = float("nan")
            if self._device.type == "cuda":
                try:
                    vram_mb = torch.cuda.memory_allocated(self._device) / (1024.0**2)
                except Exception:
                    vram_mb = float("nan")

            try:
                cpu_pct = float(self._process.cpu_percent())
            except Exception:
                cpu_pct = float("nan")

            gpu_util_pct = float("nan")
            if self._nvml_handle is not None and _pynvml is not None:
                try:
                    util = _pynvml.nvmlDeviceGetUtilizationRates(self._nvml_handle)
                    gpu_util_pct = float(util.gpu)
                except Exception:
                    gpu_util_pct = float("nan")

            sample = ResourceSample(
                dataset=self._dataset_name,
                elapsed_s=elapsed_s,
                ram_mb=float(ram_mb),
                vram_mb=float(vram_mb),
                cpu_pct=cpu_pct,
                gpu_util_pct=gpu_util_pct,
            )
            with self._lock:
                self._samples.append(sample)
