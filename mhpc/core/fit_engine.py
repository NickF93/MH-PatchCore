"""Training orchestration engine for MH-PatchCore."""

from __future__ import annotations

from collections.abc import Callable, Iterable
import logging
from typing import Any, cast

import numpy as np
import torch as _torch

from .batching import normalize_batch
from .fit_engine_helpers import _create_stage_runtime_state_impl
from .locality_state_helpers import (
    count_memory_bank_references,
    normalize_memory_bank_payload,
    resolve_anomaly_scorer_memory_bank,
    validate_anomaly_scorer_memory_bank_compatibility,
    validate_memory_bank_reference_limit,
)
from .locality_runtime_helpers import (
    build_locality_context_if_required,
    infer_projector1_with_locality,
    infer_projector2_with_locality,
    infer_transform_with_locality,
    slot_locality_kwargs,
    slot_requires_locality_context,
)
from .pipeline_stage_contract import assert_stage_allowed_in_mode
from .plugins.distance.contracts import MemoryBankPayload
from .plugins.locality_context_contract import LocalityContext
from .plugins.mem_agg.contracts import AggregationRuntimeMetadata, MemAggRuntimeContext
from .train_update_context_contract import TrainUpdateContext
from mhpc.util.progress import create_progress_bar, make_progress_postfix

LOGGER = logging.getLogger(__name__)


def _create_aggregation_strategy_default(runtime_context: Any, *, model: Any) -> Any:
    """Default aggregation runtime-state constructor."""
    return _create_stage_runtime_state_impl(
        plugin=model._mem_agg_plugin,
        factory_name="create_runtime_state",
        argument_name="runtime_context",
        argument_value=runtime_context,
    )


def _create_feature_reduction_strategy_default(
    selection: Any,
    *,
    model: Any,
) -> Any:
    """Default feature-reduction strategy constructor."""
    return _create_stage_runtime_state_impl(
        plugin=model._feature_agg_plugin,
        factory_name="create_feature_reduction_strategy",
        argument_name="selection",
        argument_value=selection,
    )


# Backward-compatible test seams: tests monkeypatch these symbols to inject
# deterministic strategy stubs without altering runtime code paths.
create_aggregation_strategy: Any = _create_aggregation_strategy_default
create_feature_reduction_strategy: Any = _create_feature_reduction_strategy_default


class FitEngine:
    """Coordinate TRAIN fit semantics and keep fit-only authority off the facade."""

    def __init__(self, model: Any) -> None:
        self._model = model

    @staticmethod
    def _normalized_training_contract(value: object) -> str:
        if not isinstance(value, str):
            raise TypeError(
                "training_contract must be a string token: "
                f"type={type(value).__name__}"
            )
        normalized = value.strip().upper()
        if normalized not in {"OFFLINE", "STREAMING"}:
            raise ValueError(
                "training_contract must be one of: OFFLINE, STREAMING."
            )
        return normalized

    @staticmethod
    def _require_reiterable_streaming_data(data: Iterable[Any]) -> None:
        """Fail fast when STREAMING input is a one-shot iterator."""
        if iter(data) is data:
            raise ValueError(
                "STREAMING fit requires a re-iterable input_data object because "
                "the algorithm performs multiple passes."
            )

    @staticmethod
    def _format_fit_iteration_desc(
        *,
        base_desc: str,
        iteration_idx: int,
        total_iterations: int,
        label: str,
    ) -> str:
        """Format progress description with optional indexed iteration label."""
        if total_iterations == 1:
            return base_desc
        return f"{base_desc} ({label} {iteration_idx}/{total_iterations})"

    @staticmethod
    def _train_update_context(
        *,
        epoch_index: int,
        epoch_count: int,
        batch_index: int,
    ) -> TrainUpdateContext:
        if epoch_index <= 0 or epoch_count <= 0 or batch_index <= 0:
            raise ValueError(
                "TrainUpdateContext fields must be positive 1-based integers."
            )
        return TrainUpdateContext(
            epoch_index=int(epoch_index),
            epoch_count=int(epoch_count),
            batch_index=int(batch_index),
        )

    def _iterate_data(
        self,
        input_data: Iterable[Any],
        desc: str = "Processing data...",
    ) -> Iterable[_torch.Tensor]:
        """Iterate over heterogeneous batches using a normalized image contract."""
        with create_progress_bar(input_data, desc=desc) as data_iterator:
            processed_images = 0
            for batch_idx, batch in enumerate(data_iterator, start=1):
                normalized = normalize_batch(batch, include_targets=False)
                image_tensor = normalized.images.to(_torch.float).to(self._model._device)
                batch_size = int(image_tensor.shape[0])
                processed_images += batch_size
                data_iterator.set_postfix(
                    make_progress_postfix(
                        batch=batch_idx,
                        batch_size=batch_size,
                        images=processed_images,
                    ),
                    refresh=False,
                )
                yield image_tensor

    def _iter_repeated_fit_batches(
        self,
        *,
        input_data: Iterable[Any],
        base_desc: str,
        total_iterations: int,
        label: str,
    ) -> Iterable[tuple[int, int, _torch.Tensor]]:
        """Yield deterministic repeated batches for streaming multi-pass fits."""
        for iteration_idx in range(1, total_iterations + 1):
            iteration_desc = self._format_fit_iteration_desc(
                base_desc=base_desc,
                iteration_idx=iteration_idx,
                total_iterations=total_iterations,
                label=label,
            )
            for batch_idx, image in enumerate(
                self._iterate_data(input_data, desc=iteration_desc),
                start=1,
            ):
                yield iteration_idx, batch_idx, image

    def _fit_embed_feature_batch(self, image: _torch.Tensor) -> np.ndarray:
        """Embed one fit batch into detached numpy patch features."""
        with _torch.no_grad():
            return cast(np.ndarray, self._model._embed(image, detach=True))

    def _fit_embed_feature_batch_with_patch_shapes(
        self,
        image: _torch.Tensor,
    ) -> tuple[np.ndarray, list[list[int]]]:
        """Embed one fit batch and return patch-shape metadata from patch-align."""
        with _torch.no_grad():
            return cast(
                tuple[np.ndarray, list[list[int]]],
                self._model._embed(image, detach=True, provide_patch_shapes=True),
            )

    def _fit_embed_feature_batch_with_locality_context(
        self,
        image: _torch.Tensor,
    ) -> tuple[np.ndarray, LocalityContext | None]:
        """Embed one fit batch and attach aligned locality metadata when required."""
        features, patch_shapes = self._fit_embed_feature_batch_with_patch_shapes(image)
        locality_context = build_locality_context_if_required(
            batch_size=int(image.shape[0]),
            patch_shapes=patch_shapes,
            plugins=(
                self._model._proj1_plugin,
                self._model._transform_plugin,
                self._model._proj2_plugin,
                self._model._mem_agg_plugin,
                self._model._scoring_plugin,
            ),
        )
        return features, locality_context

    def _fit_embed_feature_batch_with_optional_locality_context(
        self,
        image: _torch.Tensor,
        *,
        plugins: tuple[object, ...],
    ) -> tuple[np.ndarray, LocalityContext | None]:
        """Skip patch-shape collection when all requested fit-time plugins are global."""
        if not any(slot_requires_locality_context(plugin=plugin) for plugin in plugins):
            return self._fit_embed_feature_batch(image), None
        return self._fit_embed_feature_batch_with_locality_context(image)

    def _fit_create_feature_reduction_strategy(
        self,
        *,
        create_feature_reduction_strategy: Callable[..., Any],
        default_feature_reduction_strategy_factory: Callable[..., Any],
    ) -> Any:
        """Create feature-reduction strategy via plugin selection contract."""
        selection = self._model._feature_agg_plugin.resolve_reduction_selection(
            training_contract=self._model._training_contract,
        )
        self._model._feature_agg_requires_fit_state = bool(
            self._model._feature_agg_plugin.requires_fit_state(
                selection=selection,
            )
        )
        if create_feature_reduction_strategy is default_feature_reduction_strategy_factory:
            return create_feature_reduction_strategy(
                selection,
                model=self._model,
            )
        return create_feature_reduction_strategy(selection)

    def _fit_create_memory_aggregation_runtime_state(
        self,
        *,
        feature_count: int | None,
        create_aggregation_strategy: Callable[..., Any],
        default_aggregation_strategy_factory: Callable[..., Any],
    ) -> tuple[AggregationRuntimeMetadata, Any]:
        """Create memory-aggregation runtime state via plugin selection contract."""
        runtime_context = MemAggRuntimeContext(
            training_contract=self._model._training_contract,
            device=self._model._device,
            feature_count=feature_count,
        )
        if create_aggregation_strategy is not default_aggregation_strategy_factory:
            runtime_state = create_aggregation_strategy(runtime_context)
        else:
            runtime_state = create_aggregation_strategy(runtime_context, model=self._model)
        return runtime_state.runtime_metadata(), runtime_state

    def _fit_apply_projector1_features(
        self,
        *,
        features: np.ndarray,
        stage: str,
        batch_idx: int | None = None,
        locality_context: LocalityContext | None = None,
    ) -> np.ndarray:
        """Apply inference-time proj1 state during fit-time downstream phases."""
        if not self._model._uses_proj1_state():
            return np.asarray(features)
        return np.asarray(
            infer_projector1_with_locality(
                projector_plugin=self._model._proj1_plugin,
                features=np.asarray(features),
                stage=stage,
                batch_idx=batch_idx,
                locality_context=locality_context,
            )
        )

    def _fit_apply_transform_features(
        self,
        *,
        features: np.ndarray,
        stage: str,
        batch_idx: int | None = None,
        locality_context: LocalityContext | None = None,
    ) -> np.ndarray:
        """Apply inference-time transform state during fit-time downstream phases."""
        if not self._model._uses_transform_state():
            return np.asarray(features)
        return np.asarray(
            infer_transform_with_locality(
                transform_plugin=self._model._transform_plugin,
                features=np.asarray(features, dtype=np.float64),
                stage=stage,
                batch_idx=batch_idx,
                locality_context=locality_context,
            )
        )

    def _fit_apply_projector2_features(
        self,
        *,
        features: np.ndarray,
        stage: str,
        batch_idx: int | None = None,
        locality_context: LocalityContext | None = None,
    ) -> np.ndarray:
        """Apply inference-time proj2 state during fit-time downstream phases."""
        if not self._model._uses_proj2_state():
            return np.asarray(features)
        return np.asarray(
            infer_projector2_with_locality(
                projector_plugin=self._model._proj2_plugin,
                features=np.asarray(features),
                stage=stage,
                batch_idx=batch_idx,
                locality_context=locality_context,
            )
        )

    def _fit_prepare_streaming_features(
        self,
        *,
        features: np.ndarray,
        reduction_strategy: Any,
        stage: str,
        batch_idx: int | None = None,
        locality_context: LocalityContext | None = None,
        apply_proj1: bool = True,
        apply_transform: bool = True,
        apply_proj2: bool = True,
    ) -> np.ndarray:
        """Apply fit-time trainable stages in canonical deterministic order."""
        transformed = np.asarray(features)
        transformed = reduction_strategy.transform(transformed)
        if apply_proj1:
            transformed = self._fit_apply_projector1_features(
                features=transformed,
                stage=stage,
                batch_idx=batch_idx,
                locality_context=locality_context,
            )
        if apply_transform:
            transformed = self._fit_apply_transform_features(
                features=transformed,
                stage=stage,
                batch_idx=batch_idx,
                locality_context=locality_context,
            )
        if apply_proj2:
            transformed = self._fit_apply_projector2_features(
                features=transformed,
                stage=stage,
                batch_idx=batch_idx,
                locality_context=locality_context,
            )
        return np.asarray(transformed)

    def _normalize_streaming_mem_agg_handoff(
        self,
        *,
        features: np.ndarray,
    ) -> np.ndarray:
        """Preserve legacy global float64 transform output before memory aggregation."""
        handoff = np.asarray(features)
        if (
            self._model._uses_transform_state()
            and not self._model._uses_proj2_state()
            and not slot_requires_locality_context(plugin=self._model._transform_plugin)
            and not slot_requires_locality_context(plugin=self._model._mem_agg_plugin)
        ):
            handoff = np.asarray(handoff, dtype=np.float64)
        self._model._validate_finite_feature_array(
            handoff,
            stage="streaming_clustering_handoff",
        )
        return np.asarray(handoff)

    def _fit_patch_scoring_state_streaming(
        self,
        *,
        input_data: Iterable[Any],
        reduction_strategy: Any,
    ) -> None:
        """Fit patch-scoring auxiliary state after streaming memory-bank fit."""
        if not self._model._requires_patch_scoring_state():
            self._model._set_stage_owned_state(
                stage_name="scoring",
                key="aux_state",
                value=None,
            )
            return
        LOGGER.info("Phase 5: Fitting scoring auxiliary state...")
        fit_state = self._model._scoring_plugin.aux_state_fit_start(
            memory_bank=cast(
                MemoryBankPayload,
                getattr(self._model.anomaly_scorer, "detection_features", None),
            ),
        )
        for _, score_batch_idx, image in self._iter_repeated_fit_batches(
            input_data=input_data,
            base_desc="Phase 5: Streaming Scoring Fit...",
            total_iterations=1,
            label="pass",
        ):
            features, locality_context = self._fit_embed_feature_batch_with_optional_locality_context(
                image,
                plugins=(
                    self._model._proj1_plugin,
                    self._model._transform_plugin,
                    self._model._proj2_plugin,
                    self._model._scoring_plugin,
                ),
            )
            features = self._fit_prepare_streaming_features(
                features=features,
                reduction_strategy=reduction_strategy,
                stage="streaming_scoring_fit",
                batch_idx=score_batch_idx,
                locality_context=locality_context,
            )
            self._model._validate_finite_feature_array(
                features,
                stage="streaming_scoring_fit",
                batch_idx=score_batch_idx,
            )
            patch_shape = (
                tuple(locality_context.patch_shape)
                if locality_context is not None
                else (1, int(features.shape[0]))
            )
            fit_state = self._model._scoring_plugin.aux_state_fit_update(
                fit_state=fit_state,
                features=np.asarray(features, dtype=np.float32),
                batch_size=int(image.shape[0]),
                patch_shape=(int(patch_shape[0]), int(patch_shape[1])),
                **slot_locality_kwargs(
                    plugin=self._model._scoring_plugin,
                    locality_context=locality_context,
                ),
            )
        scoring_aux_state = self._model._scoring_plugin.aux_state_fit_finalize(
            fit_state=fit_state,
        )
        self._model._set_stage_owned_state(
            stage_name="scoring",
            key="aux_state",
            value=scoring_aux_state,
        )
        LOGGER.info("Scoring auxiliary state fitted.")

    def _fit_mode_offline(
        self,
        *,
        input_data: Iterable[Any],
        create_feature_reduction_strategy: Callable[..., Any],
        default_feature_reduction_strategy_factory: Callable[..., Any],
        create_aggregation_strategy: Callable[..., Any],
        default_aggregation_strategy_factory: Callable[..., Any],
    ) -> None:
        """Execute OFFLINE fit mode while preserving existing numerical behavior."""
        LOGGER.info("Using VANILLA mode (batch processing)...")
        self._model.forward_modules.eval()
        self._model._backbone.eval()

        feature_batches: list[np.ndarray] = []
        feature_batch_layouts: list[tuple[int, LocalityContext | None]] = []
        total_vectors = 0
        requires_offline_batch_layouts = (
            slot_requires_locality_context(plugin=self._model._proj1_plugin)
            or slot_requires_locality_context(plugin=self._model._proj2_plugin)
            or
            slot_requires_locality_context(plugin=self._model._transform_plugin)
            or slot_requires_locality_context(plugin=self._model._mem_agg_plugin)
        )
        with create_progress_bar(
            input_data,
            desc="Phase 1: Feature Extraction (Batch)...",
            position=1,
        ) as data_iterator:
            for batch_idx, batch in enumerate(data_iterator, start=1):
                normalized = normalize_batch(batch, include_targets=False)
                with _torch.no_grad():
                    input_image = normalized.images.to(_torch.float).to(self._model._device)
                    if requires_offline_batch_layouts:
                        batch_features, locality_context = (
                            self._fit_embed_feature_batch_with_locality_context(input_image)
                        )
                        feature_batch_layouts.append(
                            (int(batch_features.shape[0]), locality_context)
                        )
                    else:
                        batch_features = cast(
                            np.ndarray,
                            self._model._embed(input_image, detach=True),
                        )
                    feature_batches.append(batch_features)
                    total_vectors += int(batch_features.shape[0])
                    data_iterator.set_postfix(
                        make_progress_postfix(
                            batch=batch_idx,
                            batch_size=int(batch_features.shape[0]),
                            images=total_vectors,
                            phase="feature_extraction",
                        ),
                        refresh=False,
                    )

        features = np.concatenate(feature_batches, axis=0)
        self._model._validate_finite_feature_array(
            features,
            stage="vanilla_feature_extraction",
        )
        LOGGER.info(
            "Extracted %d feature vectors of dimension %d.",
            int(features.shape[0]),
            int(features.shape[1]),
        )

        reduction_strategy = self._fit_create_feature_reduction_strategy(
            create_feature_reduction_strategy=create_feature_reduction_strategy,
            default_feature_reduction_strategy_factory=default_feature_reduction_strategy_factory,
        )
        LOGGER.info("Phase 2: Fitting feature reduction (batch mode)...")
        features = reduction_strategy.fit_transform(features)
        self._model._set_stage_owned_state(
            stage_name="feature_agg",
            key="opaque_state",
            value=cast(Any, reduction_strategy.export_state()),
        )
        LOGGER.info(
            "Feature reduction output dimension: %d.",
            int(features.shape[1]),
        )

        if self._model._uses_proj1_state():
            projector1_passes = int(self._model._pipeline_stage_fit_epochs["proj1"])
            LOGGER.info(
                "Phase 3: Fitting proj1 state (batch mode, passes=%d)...",
                projector1_passes,
            )
            context = self._model._proj1_plugin.resolve_train_context(
                training_contract="OFFLINE",
                feature_dim=int(features.shape[1]),
                device=self._model._device,
            )
            self._model._proj1_plugin.train_start(context=context)
            if slot_requires_locality_context(plugin=self._model._proj1_plugin):
                for proj1_epoch_idx in range(1, projector1_passes + 1):
                    offset = 0
                    for update_batch_idx, (feature_count, locality_context) in enumerate(
                        feature_batch_layouts,
                        start=1,
                    ):
                        next_offset = offset + int(feature_count)
                        self._model._proj1_plugin.train_update(
                            batch=np.asarray(features[offset:next_offset]),
                            update_context=self._train_update_context(
                                epoch_index=proj1_epoch_idx,
                                epoch_count=projector1_passes,
                                batch_index=update_batch_idx,
                            ),
                            **slot_locality_kwargs(
                                plugin=self._model._proj1_plugin,
                                locality_context=locality_context,
                            ),
                        )
                        offset = next_offset
            else:
                for proj1_epoch_idx in range(1, projector1_passes + 1):
                    self._model._proj1_plugin.train_update(
                        batch=np.asarray(features),
                        update_context=self._train_update_context(
                            epoch_index=proj1_epoch_idx,
                            epoch_count=projector1_passes,
                            batch_index=1,
                        ),
                    )
            self._model._proj1_plugin.train_finalize()
            self._model._set_stage_owned_state(
                stage_name="proj1",
                key="opaque_state",
                value=self._model._proj1_plugin.state_export(),
            )
            if slot_requires_locality_context(plugin=self._model._proj1_plugin):
                offset = 0
                projected_batches: list[np.ndarray] = []
                for feature_count, locality_context in feature_batch_layouts:
                    next_offset = offset + int(feature_count)
                    projected_batches.append(
                        self._fit_apply_projector1_features(
                            features=np.asarray(features[offset:next_offset]),
                            stage="vanilla_fit_proj1",
                            locality_context=locality_context,
                        )
                    )
                    offset = next_offset
                features = np.concatenate(projected_batches, axis=0)
            else:
                features = self._fit_apply_projector1_features(
                    features=np.asarray(features),
                    stage="vanilla_fit_proj1",
                    locality_context=None,
                )
            LOGGER.info("Applied fitted proj1 state on training features.")
        else:
            self._model._proj1_plugin.state_load(state=None)
            self._model._set_stage_owned_state(
                stage_name="proj1",
                key="opaque_state",
                value=None,
            )

        if self._model._uses_transform_state():
            LOGGER.info("Phase 4: Fitting transform state (batch mode)...")
            context = self._model._transform_plugin.resolve_train_context(
                training_contract="OFFLINE",
                feature_dim=int(features.shape[1]),
            )
            self._model._transform_plugin.train_start(context=context)
            if slot_requires_locality_context(plugin=self._model._transform_plugin):
                offset = 0
                for update_batch_idx, (feature_count, locality_context) in enumerate(
                    feature_batch_layouts,
                    start=1,
                ):
                    next_offset = offset + int(feature_count)
                    self._model._transform_plugin.train_update(
                        batch=np.asarray(features[offset:next_offset], dtype=np.float64),
                        update_context=self._train_update_context(
                            epoch_index=1,
                            epoch_count=1,
                            batch_index=update_batch_idx,
                        ),
                        **slot_locality_kwargs(
                            plugin=self._model._transform_plugin,
                            locality_context=locality_context,
                        ),
                    )
                    offset = next_offset
            else:
                self._model._transform_plugin.train_update(
                    batch=np.asarray(features, dtype=np.float64),
                    update_context=self._train_update_context(
                        epoch_index=1,
                        epoch_count=1,
                        batch_index=1,
                    ),
                )
            self._model._transform_plugin.train_finalize()
            self._model._set_stage_owned_state(
                stage_name="transform",
                key="opaque_state",
                value=self._model._transform_plugin.state_export(),
            )
            if slot_requires_locality_context(plugin=self._model._transform_plugin):
                offset = 0
                transformed_batches: list[np.ndarray] = []
                for feature_count, locality_context in feature_batch_layouts:
                    next_offset = offset + int(feature_count)
                    transformed_batches.append(
                        infer_transform_with_locality(
                            transform_plugin=self._model._transform_plugin,
                            features=np.asarray(features[offset:next_offset], dtype=np.float64),
                            stage="vanilla_fit_transform",
                            locality_context=locality_context,
                        ).astype(np.float32, copy=False)
                    )
                    offset = next_offset
                features = np.concatenate(transformed_batches, axis=0)
            else:
                features = infer_transform_with_locality(
                    transform_plugin=self._model._transform_plugin,
                    features=np.asarray(features, dtype=np.float64),
                    stage="vanilla_fit_transform",
                    locality_context=None,
                ).astype(np.float32, copy=False)
            LOGGER.info("Applied fitted transform state on training features.")
        else:
            self._model._transform_plugin.state_load(state=None)
            self._model._set_stage_owned_state(
                stage_name="transform",
                key="opaque_state",
                value=None,
            )

        if self._model._uses_proj2_state():
            LOGGER.info("Phase 5: Fitting proj2 state (batch mode)...")
            context = self._model._proj2_plugin.resolve_train_context(
                training_contract="OFFLINE",
                feature_dim=int(features.shape[1]),
                device=self._model._device,
            )
            self._model._proj2_plugin.train_start(context=context)
            if slot_requires_locality_context(plugin=self._model._proj2_plugin):
                offset = 0
                for update_batch_idx, (feature_count, locality_context) in enumerate(
                    feature_batch_layouts,
                    start=1,
                ):
                    next_offset = offset + int(feature_count)
                    self._model._proj2_plugin.train_update(
                        batch=np.asarray(features[offset:next_offset]),
                        update_context=self._train_update_context(
                            epoch_index=1,
                            epoch_count=1,
                            batch_index=update_batch_idx,
                        ),
                        **slot_locality_kwargs(
                            plugin=self._model._proj2_plugin,
                            locality_context=locality_context,
                        ),
                    )
                    offset = next_offset
            else:
                self._model._proj2_plugin.train_update(
                    batch=np.asarray(features),
                    update_context=self._train_update_context(
                        epoch_index=1,
                        epoch_count=1,
                        batch_index=1,
                    ),
                )
            self._model._proj2_plugin.train_finalize()
            self._model._set_stage_owned_state(
                stage_name="proj2",
                key="opaque_state",
                value=self._model._proj2_plugin.state_export(),
            )
            if slot_requires_locality_context(plugin=self._model._proj2_plugin):
                offset = 0
                projected_batches = []
                for feature_count, locality_context in feature_batch_layouts:
                    next_offset = offset + int(feature_count)
                    projected_batches.append(
                        self._fit_apply_projector2_features(
                            features=np.asarray(features[offset:next_offset]),
                            stage="vanilla_fit_proj2",
                            locality_context=locality_context,
                        )
                    )
                    offset = next_offset
                features = np.concatenate(projected_batches, axis=0)
            else:
                features = self._fit_apply_projector2_features(
                    features=np.asarray(features),
                    stage="vanilla_fit_proj2",
                    locality_context=None,
                )
            LOGGER.info("Applied fitted proj2 state on training features.")
        else:
            self._model._proj2_plugin.state_load(state=None)
            self._model._set_stage_owned_state(
                stage_name="proj2",
                key="opaque_state",
                value=None,
            )

        LOGGER.info("Phase 6: Running memory aggregation (batch mode)...")
        _, aggregation_strategy = self._fit_create_memory_aggregation_runtime_state(
            feature_count=int(features.shape[0]),
            create_aggregation_strategy=create_aggregation_strategy,
            default_aggregation_strategy_factory=default_aggregation_strategy_factory,
        )
        if slot_requires_locality_context(plugin=self._model._mem_agg_plugin):
            offset = 0
            for update_batch_idx, (feature_count, locality_context) in enumerate(
                feature_batch_layouts,
                start=1,
            ):
                next_offset = offset + int(feature_count)
                aggregation_strategy.update(
                    np.asarray(features[offset:next_offset]),
                    update_context=self._train_update_context(
                        epoch_index=1,
                        epoch_count=1,
                        batch_index=update_batch_idx,
                    ),
                    **slot_locality_kwargs(
                        plugin=self._model._mem_agg_plugin,
                        locality_context=locality_context,
                    ),
                )
                offset = next_offset
        else:
            aggregation_strategy.update(
                features,
                update_context=self._train_update_context(
                    epoch_index=1,
                    epoch_count=1,
                    batch_index=1,
                ),
                **slot_locality_kwargs(
                    plugin=self._model._mem_agg_plugin,
                    locality_context=None,
                ),
            )
        self._model._set_stage_owned_state(
            stage_name="mem_agg",
            key="opaque_state",
            value=aggregation_strategy.export_state(),
        )
        assert_stage_allowed_in_mode(
            "materialize",
            execution_mode="train",
            stage_mode_capabilities=self._model._pipeline_stage_mode_capabilities,
        )
        memory_bank, model_state = self._model._materialize_plugin.materialize(
            state=aggregation_strategy,
            **slot_locality_kwargs(
                plugin=self._model._materialize_plugin,
                locality_context=None,
            ),
        )
        self._model._set_stage_owned_state(
            stage_name="materialize",
            key="opaque_state",
            value=dict(model_state),
        )

        memory_bank = normalize_memory_bank_payload(memory_bank)
        validate_anomaly_scorer_memory_bank_compatibility(
            anomaly_scorer=self._model.anomaly_scorer,
            memory_bank=memory_bank,
            stage="fit",
        )
        self._model.anomaly_scorer.fit(detection_features=[memory_bank])
        resolve_anomaly_scorer_memory_bank(
            anomaly_scorer=self._model.anomaly_scorer,
            expected_memory_bank=memory_bank,
            stage="fit",
        )
        LOGGER.info(
            "Memory bank built with %d reference features.",
            count_memory_bank_references(memory_bank),
        )

    def _fit_mode_streaming(
        self,
        *,
        input_data: Iterable[Any],
        create_feature_reduction_strategy: Callable[..., Any],
        default_feature_reduction_strategy_factory: Callable[..., Any],
        create_aggregation_strategy: Callable[..., Any],
        default_aggregation_strategy_factory: Callable[..., Any],
    ) -> None:
        """Execute STREAMING fit mode while preserving existing numerical behavior."""
        LOGGER.info("Using STREAMING mode (streaming processing)...")
        self._model.forward_modules.eval()
        self._model._backbone.eval()

        reduction_strategy = self._fit_create_feature_reduction_strategy(
            create_feature_reduction_strategy=create_feature_reduction_strategy,
            default_feature_reduction_strategy_factory=default_feature_reduction_strategy_factory,
        )

        if reduction_strategy.requires_streaming_pass:
            reduction_passes = int(self._model._pipeline_stage_fit_epochs["feature_agg"])
            if reduction_passes > 1 and not bool(
                getattr(reduction_strategy, "supports_multi_pass", False)
            ):
                strategy_name = getattr(
                    reduction_strategy,
                    "name",
                    reduction_strategy.__class__.__name__,
                )
                raise RuntimeError(
                    "Configured reduction_passes > 1, but selected feature-reduction "
                    f"strategy does not support multi-pass updates: {strategy_name}."
                )
            LOGGER.info(
                "Phase 2: Fitting feature reduction (passes=%d)...",
                reduction_passes,
            )
            for reduction_epoch_idx, pca_batch_idx, image in self._iter_repeated_fit_batches(
                input_data=input_data,
                base_desc="Phase 2: Streaming Feature Reduction...",
                total_iterations=reduction_passes,
                label="pass",
            ):
                features = self._fit_embed_feature_batch(image)
                self._model._validate_finite_feature_array(
                    features,
                    stage="streaming_pca",
                    batch_idx=pca_batch_idx,
                )
                reduction_strategy.update(
                    features,
                    update_context=self._train_update_context(
                        epoch_index=reduction_epoch_idx,
                        epoch_count=reduction_passes,
                        batch_index=pca_batch_idx,
                    ),
                )
            reduction_strategy.finalize()
            feature_dim = reduction_strategy.output_dimension
            if feature_dim is None:
                raise RuntimeError(
                    "Streaming feature-reduction strategy did not expose output dimension "
                    "after finalize()."
                )
            LOGGER.info("Feature reduction selected %d components.", int(feature_dim))
        else:
            first_batch = next(
                iter(
                    self._iterate_data(
                        input_data,
                        desc="Phase 2: Inferring feature dim...",
                    )
                )
            )
            features = self._fit_embed_feature_batch(first_batch)
            transformed = reduction_strategy.transform(features)
            feature_dim = int(transformed.shape[1])

        self._model._set_stage_owned_state(
            stage_name="feature_agg",
            key="opaque_state",
            value=cast(Any, reduction_strategy.export_state()),
        )

        if self._model._uses_proj1_state():
            projector1_passes = int(self._model._pipeline_stage_fit_epochs["proj1"])
            LOGGER.info(
                "Phase 3: Fitting proj1 state (passes=%d)...",
                projector1_passes,
            )
            context = self._model._proj1_plugin.resolve_train_context(
                training_contract="STREAMING",
                feature_dim=int(feature_dim),
                device=self._model._device,
            )
            self._model._proj1_plugin.train_start(context=context)

            for proj1_epoch_idx, proj1_batch_idx, image in self._iter_repeated_fit_batches(
                input_data=input_data,
                base_desc="Phase 3: Streaming Proj1...",
                total_iterations=projector1_passes,
                label="pass",
            ):
                features, locality_context = (
                    self._fit_embed_feature_batch_with_optional_locality_context(
                        image,
                        plugins=(self._model._proj1_plugin,),
                    )
                )
                features = reduction_strategy.transform(features)
                self._model._validate_finite_feature_array(
                    features,
                    stage="streaming_proj1",
                    batch_idx=proj1_batch_idx,
                )
                self._model._proj1_plugin.train_update(
                    batch=np.asarray(features),
                    update_context=self._train_update_context(
                        epoch_index=proj1_epoch_idx,
                        epoch_count=projector1_passes,
                        batch_index=proj1_batch_idx,
                    ),
                    **slot_locality_kwargs(
                        plugin=self._model._proj1_plugin,
                        locality_context=locality_context,
                    ),
                )

            self._model._proj1_plugin.train_finalize()
            self._model._set_stage_owned_state(
                stage_name="proj1",
                key="opaque_state",
                value=self._model._proj1_plugin.state_export(),
            )
            LOGGER.info("Computed proj1-stage fit state.")
        else:
            self._model._proj1_plugin.state_load(state=None)
            self._model._set_stage_owned_state(
                stage_name="proj1",
                key="opaque_state",
                value=None,
            )

        if self._model._uses_transform_state():
            covariance_passes = int(self._model._pipeline_stage_fit_epochs["transform"])
            LOGGER.info(
                "Phase 4: Fitting transform state (passes=%d)...",
                covariance_passes,
            )
            context = self._model._transform_plugin.resolve_train_context(
                training_contract="STREAMING",
                feature_dim=int(feature_dim),
            )
            self._model._transform_plugin.train_start(context=context)

            for transform_epoch_idx, cov_batch_idx, image in self._iter_repeated_fit_batches(
                input_data=input_data,
                base_desc="Phase 3: Streaming Covariance...",
                total_iterations=covariance_passes,
                label="pass",
            ):
                features, locality_context = (
                    self._fit_embed_feature_batch_with_optional_locality_context(
                        image,
                        plugins=(self._model._proj1_plugin, self._model._transform_plugin),
                    )
                )
                features = self._fit_prepare_streaming_features(
                    features=features,
                    reduction_strategy=reduction_strategy,
                    stage="streaming_covariance",
                    batch_idx=cov_batch_idx,
                    locality_context=locality_context,
                    apply_proj2=False,
                    apply_transform=False,
                )
                self._model._validate_finite_feature_array(
                    features,
                    stage="streaming_covariance",
                    batch_idx=cov_batch_idx,
                )
                self._model._transform_plugin.train_update(
                    batch=np.asarray(features, dtype=np.float64),
                    update_context=self._train_update_context(
                        epoch_index=transform_epoch_idx,
                        epoch_count=covariance_passes,
                        batch_index=cov_batch_idx,
                    ),
                    **slot_locality_kwargs(
                        plugin=self._model._transform_plugin,
                        locality_context=locality_context,
                    ),
                )

            self._model._transform_plugin.train_finalize()
            self._model._set_stage_owned_state(
                stage_name="transform",
                key="opaque_state",
                value=self._model._transform_plugin.state_export(),
            )
            LOGGER.info("Computed transform-stage fit state.")
        else:
            self._model._transform_plugin.state_load(state=None)
            self._model._set_stage_owned_state(
                stage_name="transform",
                key="opaque_state",
                value=None,
            )

        if self._model._uses_proj2_state():
            projector2_passes = int(self._model._pipeline_stage_fit_epochs["proj2"])
            LOGGER.info(
                "Phase 5: Fitting proj2 state (passes=%d)...",
                projector2_passes,
            )
            context = self._model._proj2_plugin.resolve_train_context(
                training_contract="STREAMING",
                feature_dim=int(feature_dim),
                device=self._model._device,
            )
            self._model._proj2_plugin.train_start(context=context)

            for proj2_epoch_idx, proj2_batch_idx, image in self._iter_repeated_fit_batches(
                input_data=input_data,
                base_desc="Phase 5: Streaming Proj2...",
                total_iterations=projector2_passes,
                label="pass",
            ):
                features, locality_context = (
                    self._fit_embed_feature_batch_with_optional_locality_context(
                        image,
                        plugins=(
                            self._model._proj1_plugin,
                            self._model._transform_plugin,
                            self._model._proj2_plugin,
                        ),
                    )
                )
                features = self._fit_prepare_streaming_features(
                    features=features,
                    reduction_strategy=reduction_strategy,
                    stage="streaming_proj2",
                    batch_idx=proj2_batch_idx,
                    locality_context=locality_context,
                    apply_proj2=False,
                )
                self._model._validate_finite_feature_array(
                    features,
                    stage="streaming_proj2",
                    batch_idx=proj2_batch_idx,
                )
                self._model._proj2_plugin.train_update(
                    batch=np.asarray(features),
                    update_context=self._train_update_context(
                        epoch_index=proj2_epoch_idx,
                        epoch_count=projector2_passes,
                        batch_index=proj2_batch_idx,
                    ),
                    **slot_locality_kwargs(
                        plugin=self._model._proj2_plugin,
                        locality_context=locality_context,
                    ),
                )

            self._model._proj2_plugin.train_finalize()
            self._model._set_stage_owned_state(
                stage_name="proj2",
                key="opaque_state",
                value=self._model._proj2_plugin.state_export(),
            )
            LOGGER.info("Computed proj2-stage fit state.")
        else:
            self._model._proj2_plugin.state_load(state=None)
            self._model._set_stage_owned_state(
                stage_name="proj2",
                key="opaque_state",
                value=None,
            )

        aggregation_runtime_metadata, aggregation_strategy = (
            self._fit_create_memory_aggregation_runtime_state(
                feature_count=None,
                create_aggregation_strategy=create_aggregation_strategy,
                default_aggregation_strategy_factory=default_aggregation_strategy_factory,
            )
        )
        LOGGER.info("Phase 6: Running memory aggregation (streaming mode)...")
        aggregation_passes = int(self._model._pipeline_stage_fit_epochs["mem_agg"])
        if aggregation_passes > 1 and not bool(
            getattr(aggregation_strategy, "supports_multi_pass", False)
        ):
            strategy_name = getattr(
                aggregation_strategy,
                "name",
                aggregation_strategy.__class__.__name__,
            )
            raise RuntimeError(
                "Configured aggregation_passes > 1, but selected aggregation strategy "
                f"does not support multi-pass updates: {strategy_name}."
            )
        LOGGER.info(
            "Phase 6: Streaming aggregation configured with passes=%d.",
            aggregation_passes,
        )

        for aggregation_epoch_idx, aggregation_batch_idx, image in self._iter_repeated_fit_batches(
            input_data=input_data,
            base_desc="Phase 6: Streaming Aggregation...",
            total_iterations=aggregation_passes,
            label="pass",
        ):
            features, locality_context = (
                self._fit_embed_feature_batch_with_optional_locality_context(
                    image,
                    plugins=(
                        self._model._proj1_plugin,
                        self._model._transform_plugin,
                        self._model._proj2_plugin,
                        self._model._mem_agg_plugin,
                    ),
                )
            )
            features = self._fit_prepare_streaming_features(
                features=features,
                reduction_strategy=reduction_strategy,
                stage="streaming_clustering",
                locality_context=locality_context,
            )
            features = self._normalize_streaming_mem_agg_handoff(
                features=features,
            )
            aggregation_strategy.update(
                features,
                update_context=self._train_update_context(
                    epoch_index=aggregation_epoch_idx,
                    epoch_count=aggregation_passes,
                    batch_index=aggregation_batch_idx,
                ),
                **slot_locality_kwargs(
                    plugin=self._model._mem_agg_plugin,
                    locality_context=locality_context,
                ),
            )

        self._model._set_stage_owned_state(
            stage_name="mem_agg",
            key="opaque_state",
            value=aggregation_strategy.export_state(),
        )
        assert_stage_allowed_in_mode(
            "materialize",
            execution_mode="train",
            stage_mode_capabilities=self._model._pipeline_stage_mode_capabilities,
        )
        memory_bank, model_state = self._model._materialize_plugin.materialize(
            state=aggregation_strategy,
            **slot_locality_kwargs(
                plugin=self._model._materialize_plugin,
                locality_context=None,
            ),
        )
        self._model._set_stage_owned_state(
            stage_name="materialize",
            key="opaque_state",
            value=dict(model_state),
        )
        materialized_reference_count = validate_memory_bank_reference_limit(
            memory_bank=memory_bank,
            reference_limit=aggregation_runtime_metadata.reference_limit,
            enforce_reference_limit=bool(
                aggregation_runtime_metadata.enforce_reference_limit
            ),
            stage="Streaming aggregation",
        )
        if (
            aggregation_runtime_metadata.reference_limit is not None
            and materialized_reference_count
            > int(aggregation_runtime_metadata.reference_limit)
        ):
            if bool(aggregation_runtime_metadata.enforce_reference_limit):
                raise AssertionError(
                    "validate_memory_bank_reference_limit returned a limit overflow "
                    "without raising while enforcement was enabled."
                )
            LOGGER.warning(
                "Streaming aggregation exceeded configured reference budget=%d "
                "(got %d) "
                "but enforcement is disabled by configuration "
                "(streaming_enforce_cluster_budget=false).",
                int(aggregation_runtime_metadata.reference_limit),
                materialized_reference_count,
            )
        memory_bank = normalize_memory_bank_payload(memory_bank)
        validate_anomaly_scorer_memory_bank_compatibility(
            anomaly_scorer=self._model.anomaly_scorer,
            memory_bank=memory_bank,
            stage="fit",
        )
        LOGGER.info(
            "Aggregation produced %d memory vectors.",
            count_memory_bank_references(memory_bank),
        )
        self._model.anomaly_scorer.fit(detection_features=[memory_bank])
        resolve_anomaly_scorer_memory_bank(
            anomaly_scorer=self._model.anomaly_scorer,
            expected_memory_bank=memory_bank,
            stage="fit",
        )
        LOGGER.info(
            "Memory bank built with %d reference features.",
            count_memory_bank_references(memory_bank),
        )
        self._fit_patch_scoring_state_streaming(
            input_data=input_data,
            reduction_strategy=reduction_strategy,
        )

    def _run_fit_mode(
        self,
        *,
        mode: str,
        input_data: Iterable[Any],
    ) -> None:
        """Run one fit mode via FitEngine-owned semantic mode hook."""
        mode_hooks: dict[str, str] = {
            "OFFLINE": "_fit_mode_offline",
            "STREAMING": "_fit_mode_streaming",
        }
        hook_name = mode_hooks.get(mode)
        if hook_name is None:
            raise RuntimeError(f"Unsupported fit mode: {mode!r}")
        hook = getattr(self, hook_name, None)
        if not callable(hook):
            raise TypeError(
                "FitEngine is missing required fit-mode hook: "
                f"name='{hook_name}' type='{type(hook).__name__}'"
            )
        hook(
            input_data=input_data,
            create_feature_reduction_strategy=create_feature_reduction_strategy,
            default_feature_reduction_strategy_factory=_create_feature_reduction_strategy_default,
            create_aggregation_strategy=create_aggregation_strategy,
            default_aggregation_strategy_factory=_create_aggregation_strategy_default,
        )

    def fit(self, data: Iterable[Any]) -> Any:
        """Run TRAIN fit orchestration and delegate mode execution."""
        training_contract = self._normalized_training_contract(
            self._model._training_contract
        )
        if training_contract == "STREAMING":
            self._require_reiterable_streaming_data(data)
        self._run_fit_mode(
            mode=training_contract,
            input_data=data,
        )
        return self._model

    def fill_memory_bank_vanilla(self, input_data: Iterable[Any]) -> None:
        """Run OFFLINE fit-mode through FitEngine-owned semantics."""
        self._run_fit_mode(
            mode="OFFLINE",
            input_data=input_data,
        )

    def fill_memory_bank_streaming(self, input_data: Iterable[Any]) -> None:
        """Run STREAMING fit-mode through FitEngine-owned semantics."""
        self._run_fit_mode(
            mode="STREAMING",
            input_data=input_data,
        )
