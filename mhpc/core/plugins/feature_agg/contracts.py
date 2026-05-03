"""Slot-local contracts for feature-aggregation plugins."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np
import torch
from mhpc.core.train_update_context_contract import TrainUpdateContext


@runtime_checkable
class FeatureAggBindContextLike(Protocol):
    """Protocol-only bind context surface consumed by feature-agg plugins."""

    training_contract: str
    seed: int


@dataclass(frozen=True)
class ReductionSelection:
    """Reducer-selection inputs resolved from runtime configuration."""

    algorithm: str
    pca_variance_ratio: float
    component_selection_mode: str = "variance_ratio"
    component_count: int | None = None
    rff_n_components: int = 2048
    rff_gamma: str | float = "auto"
    rff_random_state: int = 0
    ae_latent_dim: int = 16
    ae_hidden_dims: tuple[int, ...] = ()
    ae_activation: str = "relu"
    ae_optimizer: str = "adamw"
    ae_learning_rate: float = 1.0e-3
    ae_weight_decay: float = 1.0e-4
    ae_sgd_momentum: float = 0.9
    ae_updates_per_batch: int = 1
    ae_lr_decay_policy: str = "none"
    ae_lr_decay_step_size: int = 200
    ae_lr_decay_gamma: float = 0.5
    ae_lr_decay_t_max: int = 1000


@runtime_checkable
class FeatureReductionStrategy(Protocol):
    """Contract-only protocol for fit-time feature-reduction strategies."""

    @property
    def name(self) -> str:
        """Human-readable strategy name for logs/errors."""
        ...

    @property
    def requires_streaming_pass(self) -> bool:
        """Whether this reducer needs dedicated update-pass calls."""
        ...

    @property
    def supports_multi_pass(self) -> bool:
        """Whether repeated streaming passes are supported by this reducer."""
        ...

    @property
    def output_dimension(self) -> int | None:
        """Output feature dimension when known; otherwise ``None``."""
        ...

    def update(
        self,
        batch: np.ndarray,
        update_context: TrainUpdateContext | None = None,
    ) -> None:
        """Consume a training batch for incremental/statistical updates."""
        ...

    def finalize(self) -> None:
        """Finalize state after all streaming updates are consumed."""
        ...

    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        """Fit reducer in batch mode and return transformed features."""
        ...

    def transform(self, features: np.ndarray) -> np.ndarray:
        """Transform features using fitted/finalized reducer state."""
        ...

    def export_state(self) -> Any | None:
        """Return serializable state object persisted by checkpointing."""
        ...


@runtime_checkable
class FeatureAggregatorPlugin(Protocol):
    """Contract-only protocol for STEP_4 feature aggregation behavior."""

    def bind_params(
        self,
        *,
        params: Mapping[str, Any],
        bind_context: FeatureAggBindContextLike,
    ) -> None:
        """Bind full slot parameter bundle once for this plugin instance."""
        ...

    def forward_embed_feature_aggregation(
        self,
        *,
        features: torch.Tensor,
        forward_modules: torch.nn.ModuleDict,
    ) -> torch.Tensor:
        """Run feature-aggregation forward using fixed module-key semantics."""
        ...

    def create_preadapt_aggregator_module(
        self,
    ) -> torch.nn.Module:
        """Build pre-adaptation aggregation runtime module."""
        ...

    def create_feature_reduction_strategy(
        self,
        *,
        selection: ReductionSelection,
    ) -> FeatureReductionStrategy:
        """Build fit-time feature-reduction strategy for the selected algorithm."""
        ...

    def requires_fit_state(
        self,
        *,
        selection: ReductionSelection,
    ) -> bool:
        """Return whether inference requires fitted reducer state for selection."""
        ...

    def resolve_target_embed_dimension(self) -> int:
        """Resolve target embedding dimension from plugin-bound params."""
        ...

    def resolve_reduction_selection(
        self,
        *,
        training_contract: str,
    ) -> ReductionSelection:
        """Resolve plugin-local reduction selection from bound params."""
        ...


__all__ = [
    "FeatureAggregatorPlugin",
    "FeatureAggBindContextLike",
    "FeatureReductionStrategy",
    "ReductionSelection",
    "TrainUpdateContext",
]
