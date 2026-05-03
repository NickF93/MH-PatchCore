"""Evaluation utilities for MH-PatchCore experiments."""

from __future__ import annotations

from .config import RunConfig, load_run_config
from .experiment_summary import RunSummaryRecord, summarize_experiment_root


def run_experiment(config: RunConfig):
    from .pipeline import run_experiment as _run_experiment

    return _run_experiment(config)

__all__ = [
    "RunConfig",
    "RunSummaryRecord",
    "load_run_config",
    "run_experiment",
    "summarize_experiment_root",
]
