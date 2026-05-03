"""Dedicated reproducibility preamble for experiment execution."""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch

from mhpc.util.param_binding import normalize_reproducibility_seed


@dataclass(frozen=True)
class ReproducibilityPreamble:
    """Canonical run-wide reproducibility inputs resolved before execution."""

    seed: int
    device_type: str


def build_reproducibility_preamble(
    *,
    seed: object,
    device: torch.device,
) -> ReproducibilityPreamble:
    """Resolve canonical reproducibility metadata for one run."""
    return ReproducibilityPreamble(
        seed=normalize_reproducibility_seed(seed),
        device_type=str(device.type),
    )


def apply_reproducibility_preamble(
    preamble: ReproducibilityPreamble,
) -> None:
    """Apply deterministic RNG/framework state from one preamble."""
    random.seed(preamble.seed)
    np.random.seed(preamble.seed)
    torch.manual_seed(preamble.seed)

    if preamble.device_type == "cuda":
        torch.cuda.manual_seed(preamble.seed)
        torch.cuda.manual_seed_all(preamble.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
