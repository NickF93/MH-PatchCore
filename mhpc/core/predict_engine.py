"""Inference orchestration engine for MH-PatchCore."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import torch

from .batching import normalize_batch
from .inference_output_contract import InferenceBatchOutput, as_inference_batch_output
from .locality_state_helpers import (
    flatten_distance_query_payload,
    resolve_anomaly_scorer_memory_bank,
)
from .locality_runtime_helpers import (
    build_locality_context_if_required,
    infer_projector1_with_locality,
    infer_projector2_with_locality,
    infer_transform_with_locality,
    slot_locality_kwargs,
)
from .plugins.locality_context_contract import LocalityContext
from mhpc.util.progress import (
    INFERENCE_PROGRESS_DESC,
    create_progress_bar,
    make_progress_postfix,
)

SlotOutputPayload = np.ndarray | Mapping[str, np.ndarray]


@dataclass(frozen=True)
class SlotInferenceBatchOutput:
    """Inference prediction plus selected generic slot-boundary payloads."""

    prediction: InferenceBatchOutput
    slot_outputs: Mapping[str, SlotOutputPayload]


class PredictEngine:
    """Coordinate inference orchestration and own predict-only runtime semantics."""

    def __init__(self, model: Any) -> None:
        self._model = model

    def predict(
        self,
        data: torch.utils.data.DataLoader[Any] | torch.Tensor,
    ) -> (
        tuple[list[float], list[np.ndarray]]
        | tuple[list[float], list[np.ndarray], list[int], list[np.ndarray]]
    ):
        """Run prediction for a batch tensor or a dataloader."""
        if isinstance(data, torch.utils.data.DataLoader):
            return self.predict_dataloader(data)
        normalized = normalize_batch(data, include_targets=False)
        return self.predict_batch(normalized.images)

    def predict_dataloader(
        self,
        dataloader: torch.utils.data.DataLoader[Any],
    ) -> tuple[list[float], list[np.ndarray], list[int], list[np.ndarray]]:
        """Run prediction over a dataloader and preserve labels/masks when present."""
        self._model.forward_modules.eval()
        self._model._backbone.eval()

        scores: list[float] = []
        masks: list[np.ndarray] = []
        labels_gt: list[int] = []
        masks_gt: list[np.ndarray] = []
        processed_images = 0
        anomalies_seen = 0
        with create_progress_bar(
            dataloader,
            desc=INFERENCE_PROGRESS_DESC,
        ) as data_iterator:
            for batch_idx, batch in enumerate(data_iterator, start=1):
                normalized = normalize_batch(batch, include_targets=True)
                if normalized.labels:
                    labels_gt.extend(normalized.labels)
                    anomalies_seen += int(sum(normalized.labels))
                if normalized.masks:
                    masks_gt.extend(normalized.masks)

                batch_scores, batch_masks = self.predict_batch(normalized.images)
                scores.extend(batch_scores)
                masks.extend(batch_masks)
                processed_images += len(batch_scores)
                data_iterator.set_postfix(
                    make_progress_postfix(
                        batch=batch_idx,
                        batch_size=len(batch_scores),
                        images=processed_images,
                        phase=f"anomalies={anomalies_seen}",
                    ),
                    refresh=False,
                )
        return scores, masks, labels_gt, masks_gt

    def _predict_embed_features(
        self,
        images: torch.Tensor,
    ) -> tuple[np.ndarray, list[list[int]]]:
        """Embed one inference batch and retain patch-shape metadata."""
        with torch.no_grad():
            return cast(
                tuple[np.ndarray, list[list[int]]],
                self._model._embed(images, provide_patch_shapes=True),
            )

    def _predict_reduce_features(self, features: np.ndarray) -> np.ndarray:
        """Apply fitted feature reduction state if one is required and present."""
        reducer = self._model._get_stage_owned_state(
            stage_name="feature_agg",
            key="opaque_state",
            default=None,
        )
        if reducer is None and bool(self._model._requires_feature_agg_fit_state()):
            raise RuntimeError(
                "Feature reduction is enabled but reducer state is missing or "
                "invalid. Call fit() before infer()."
            )
        if reducer is None:
            return np.asarray(features)
        if not hasattr(reducer, "transform"):
            raise RuntimeError(
                "Feature reduction is enabled but reducer state is missing or "
                "invalid. Call fit() before infer()."
            )
        return np.asarray(reducer.transform(features))

    def _predict_transform_features(
        self,
        *,
        features: np.ndarray,
        locality_context: LocalityContext | None,
    ) -> np.ndarray:
        """Apply fitted transform state if the selected transform owns one."""
        if not bool(self._model._uses_transform_state()):
            return np.asarray(features)
        return np.asarray(
            infer_transform_with_locality(
                transform_plugin=self._model._transform_plugin,
                features=np.asarray(features, dtype=np.float64),
                stage="predict",
                locality_context=locality_context,
            )
        )

    def _predict_projector1_features(
        self,
        *,
        features: np.ndarray,
        locality_context: LocalityContext | None,
    ) -> np.ndarray:
        """Apply fitted proj1 state if the selected projector owns one."""
        if not bool(self._model._uses_proj1_state()):
            return np.asarray(features)
        projector_state = self._model._get_stage_owned_state(
            stage_name="proj1",
            key="opaque_state",
            default=None,
        )
        if projector_state is None:
            raise RuntimeError(
                "Projector-1 is enabled but projector state is missing or invalid. "
                "Call fit() before infer()."
            )
        return np.asarray(
            infer_projector1_with_locality(
                projector_plugin=self._model._proj1_plugin,
                features=np.asarray(features),
                stage="predict",
                locality_context=locality_context,
            )
        )

    def _predict_projector2_features(
        self,
        *,
        features: np.ndarray,
        locality_context: LocalityContext | None,
    ) -> np.ndarray:
        """Apply fitted proj2 state if the selected projector owns one."""
        if not bool(self._model._uses_proj2_state()):
            return np.asarray(features)
        projector_state = self._model._get_stage_owned_state(
            stage_name="proj2",
            key="opaque_state",
            default=None,
        )
        if projector_state is None:
            raise RuntimeError(
                "Projector-2 is enabled but projector state is missing or invalid. "
                "Call fit() before infer()."
            )
        return np.asarray(
            infer_projector2_with_locality(
                projector_plugin=self._model._proj2_plugin,
                features=np.asarray(features),
                stage="predict",
                locality_context=locality_context,
            )
        )

    def _predict_query_distance(
        self,
        *,
        features: np.ndarray,
        batchsize: int,
        patch_shape: tuple[int, int],
        locality_context: LocalityContext | None,
    ) -> tuple[object, np.ndarray, np.ndarray, np.ndarray]:
        """Run distance query and flatten the generic query payload."""
        pre_query_memory_bank = resolve_anomaly_scorer_memory_bank(
            anomaly_scorer=self._model.anomaly_scorer,
            stage="predict pre-query",
        )
        distance_query = self._model._distance_plugin.query(
            anomaly_scorer=self._model.anomaly_scorer,
            features=features,
            **slot_locality_kwargs(
                plugin=self._model._distance_plugin,
                locality_context=locality_context,
            ),
        )
        patch_scores, query_distances, query_nns = flatten_distance_query_payload(
            distance_query,
            memory_bank=resolve_anomaly_scorer_memory_bank(
                anomaly_scorer=self._model.anomaly_scorer,
                expected_memory_bank=pre_query_memory_bank,
                stage="predict",
            ),
            batchsize=batchsize,
            patch_shape=patch_shape,
        )
        return (
            distance_query,
            np.asarray(patch_scores),
            np.asarray(query_distances),
            np.asarray(query_nns),
        )

    def _predict_score_batch(
        self,
        *,
        features: np.ndarray,
        patch_scores: np.ndarray,
        query_distances: np.ndarray,
        query_nns: np.ndarray,
        distance_query: object,
        patch_shape: tuple[int, int],
        batchsize: int,
        locality_context: LocalityContext | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Delegate final scoring to the active scoring plugin."""
        scoring_controls = self._model._scoring_plugin.resolve_scoring_controls()
        scoring_aux_state = self._model._get_stage_owned_state(
            stage_name="scoring",
            key="aux_state",
            default=None,
        )
        image_scores, effective_patch_scores = self._model._scoring_plugin.score(
            features=features,
            patch_scores=patch_scores,
            query_distances=query_distances,
            query_nns=query_nns,
            distance_query=distance_query,
            patch_shape=patch_shape,
            batchsize=batchsize,
            patch_maker=self._model.patch_maker,
            anomaly_scorer=self._model.anomaly_scorer,
            patch_scoring_mode=scoring_controls.patch_scoring_mode,
            patch_scoring_state=scoring_aux_state,
            paper_reweight_num_nn=scoring_controls.paper_reweight_num_nn,
            **slot_locality_kwargs(
                plugin=self._model._scoring_plugin,
                locality_context=locality_context,
            ),
        )
        return np.asarray(image_scores), np.asarray(effective_patch_scores)

    def predict_batch(self, images: torch.Tensor) -> tuple[list[float], list[np.ndarray]]:
        """Run one-batch inference through PredictEngine-owned batch semantics."""
        images = images.to(torch.float).to(self._model._device)
        self._model.forward_modules.eval()
        self._model._backbone.eval()

        batchsize = int(images.shape[0])
        with torch.no_grad():
            features, patch_shapes = self._predict_embed_features(images)
            patch_shape = (int(patch_shapes[0][0]), int(patch_shapes[0][1]))

            projector_locality_context = build_locality_context_if_required(
                batch_size=batchsize,
                patch_shapes=patch_shapes,
                plugins=(
                    self._model._proj1_plugin,
                    self._model._transform_plugin,
                    self._model._proj2_plugin,
                ),
            )
            distance_locality_context = build_locality_context_if_required(
                batch_size=batchsize,
                patch_shapes=patch_shapes,
                plugins=(self._model._distance_plugin,),
            )
            scoring_locality_context = build_locality_context_if_required(
                batch_size=batchsize,
                patch_shapes=patch_shapes,
                plugins=(self._model._scoring_plugin,),
            )

            features_np = self._predict_reduce_features(np.asarray(features))
            features_np = self._predict_projector1_features(
                features=features_np,
                locality_context=projector_locality_context,
            )
            features_np = self._predict_transform_features(
                features=features_np,
                locality_context=projector_locality_context,
            )
            features_np = self._predict_projector2_features(
                features=features_np,
                locality_context=projector_locality_context,
            )
            features_f32 = np.asarray(features_np).astype(np.float32, copy=False)
            distance_query, patch_scores, query_distances, query_nns = (
                self._predict_query_distance(
                    features=features_f32,
                    batchsize=batchsize,
                    patch_shape=patch_shape,
                    locality_context=distance_locality_context,
                )
            )
            image_scores, effective_patch_scores = self._predict_score_batch(
                features=features_f32,
                patch_scores=patch_scores,
                query_distances=query_distances,
                query_nns=query_nns,
                distance_query=distance_query,
                patch_shape=patch_shape,
                batchsize=batchsize,
                locality_context=scoring_locality_context,
            )

            patch_scores_tensor = torch.as_tensor(
                effective_patch_scores,
                dtype=torch.float32,
            )
            patch_score_grid = self._model.patch_maker.unpatch_scores(
                patch_scores_tensor,
                batchsize=batchsize,
            )
            patch_score_grid = patch_score_grid.reshape(
                batchsize,
                int(patch_shape[0]),
                int(patch_shape[1]),
            )
            masks = self._model.anomaly_segmentor.convert_to_segmentation(
                patch_score_grid
            )

        return [float(score) for score in image_scores], [
            np.asarray(mask) for mask in masks
        ]

    def predict_batch_with_slot_outputs(
        self,
        images: torch.Tensor,
        *,
        selected_slots: Iterable[str],
    ) -> SlotInferenceBatchOutput:
        """Run one-batch inference and collect selected generic slot payloads."""

        selected = tuple(selected_slots)
        selected_set = set(selected)
        slot_outputs: dict[str, SlotOutputPayload] = {}

        def _capture(slot_name: str, payload: SlotOutputPayload) -> None:
            if slot_name in selected_set:
                slot_outputs[slot_name] = payload

        batch_scores, batch_masks = self._predict_batch_with_capture(
            images,
            capture_slot=_capture,
        )
        missing_slots = [slot for slot in selected if slot not in slot_outputs]
        if missing_slots:
            raise RuntimeError(
                "Selected replay slots could not produce generic export payloads: "
                f"{', '.join(missing_slots)}."
            )
        return SlotInferenceBatchOutput(
            prediction=as_inference_batch_output((batch_scores, batch_masks)),
            slot_outputs=slot_outputs,
        )

    def _predict_batch_with_capture(
        self,
        images: torch.Tensor,
        *,
        capture_slot,
    ) -> tuple[list[float], list[np.ndarray]]:
        images = images.to(torch.float).to(self._model._device)
        self._model.forward_modules.eval()
        self._model._backbone.eval()

        batchsize = int(images.shape[0])
        with torch.no_grad():
            backbone_features = self._model._capture_embed_features(images)
            capture_slot(
                "backbone",
                _tensor_sequence_payload(backbone_features, batch_size=batchsize),
            )

            patch_features, patch_shapes = self._model._patchify_and_align_features(
                backbone_features
            )
            capture_slot(
                "patch_align",
                _tensor_sequence_payload(patch_features, batch_size=batchsize),
            )

            projector_locality_context = build_locality_context_if_required(
                batch_size=batchsize,
                patch_shapes=patch_shapes,
                plugins=(
                    self._model._proj1_plugin,
                    self._model._transform_plugin,
                    self._model._proj2_plugin,
                ),
            )
            distance_locality_context = build_locality_context_if_required(
                batch_size=batchsize,
                patch_shapes=patch_shapes,
                plugins=(self._model._distance_plugin,),
            )
            scoring_locality_context = build_locality_context_if_required(
                batch_size=batchsize,
                patch_shapes=patch_shapes,
                plugins=(self._model._scoring_plugin,),
            )

            processed = self._model._preprocess_plugin.forward_embed_preprocess(
                features=patch_features,
                forward_modules=self._model.forward_modules,
            )
            capture_slot(
                "preprocess",
                _batch_major_payload(_tensor_payload(processed), batchsize),
            )
            processed = (
                self._model._feature_agg_plugin.forward_embed_feature_aggregation(
                    features=processed,
                    forward_modules=self._model.forward_modules,
                )
            )
            processed = self._model._proj1_plugin.forward_embed_projector1(
                features=processed,
                forward_modules=self._model.forward_modules,
                **slot_locality_kwargs(
                    plugin=self._model._proj1_plugin,
                    locality_context=projector_locality_context,
                ),
            )
            processed = self._model._transform_plugin.forward_embed_transform(
                features=processed,
                forward_modules=self._model.forward_modules,
                **slot_locality_kwargs(
                    plugin=self._model._transform_plugin,
                    locality_context=projector_locality_context,
                ),
            )
            processed = self._model._proj2_plugin.forward_embed_projector2(
                features=processed,
                forward_modules=self._model.forward_modules,
                **slot_locality_kwargs(
                    plugin=self._model._proj2_plugin,
                    locality_context=projector_locality_context,
                ),
            )

            features_np = self._predict_reduce_features(_tensor_payload(processed))
            capture_slot(
                "feature_agg",
                _batch_major_payload(features_np, batchsize),
            )
            features_np = self._predict_projector1_features(
                features=features_np,
                locality_context=projector_locality_context,
            )
            capture_slot("proj1", _batch_major_payload(features_np, batchsize))
            features_np = self._predict_transform_features(
                features=features_np,
                locality_context=projector_locality_context,
            )
            capture_slot("transform", _batch_major_payload(features_np, batchsize))
            features_np = self._predict_projector2_features(
                features=features_np,
                locality_context=projector_locality_context,
            )
            capture_slot("proj2", _batch_major_payload(features_np, batchsize))

            features_f32 = np.asarray(features_np).astype(np.float32, copy=False)
            patch_shape = (int(patch_shapes[0][0]), int(patch_shapes[0][1]))
            distance_query, patch_scores, query_distances, query_nns = (
                self._predict_query_distance(
                    features=features_f32,
                    batchsize=batchsize,
                    patch_shape=patch_shape,
                    locality_context=distance_locality_context,
                )
            )
            capture_slot(
                "distance",
                {
                    "distance_map": np.asarray(patch_scores).reshape(batchsize, -1),
                },
            )
            image_scores, effective_patch_scores = self._predict_score_batch(
                features=features_f32,
                patch_scores=patch_scores,
                query_distances=query_distances,
                query_nns=query_nns,
                distance_query=distance_query,
                patch_shape=patch_shape,
                batchsize=batchsize,
                locality_context=scoring_locality_context,
            )
            patch_scores_tensor = torch.as_tensor(
                effective_patch_scores,
                dtype=torch.float32,
            )
            patch_score_grid = self._model.patch_maker.unpatch_scores(
                patch_scores_tensor,
                batchsize=batchsize,
            )
            patch_score_grid = patch_score_grid.reshape(
                batchsize,
                int(patch_shape[0]),
                int(patch_shape[1]),
            )
            masks = self._model.anomaly_segmentor.convert_to_segmentation(
                patch_score_grid
            )
            capture_slot(
                "scoring",
                {
                    "heatmap": np.asarray(masks).reshape(batchsize, -1),
                    "score": np.asarray(image_scores),
                },
            )

        return [float(score) for score in image_scores], [
            np.asarray(mask) for mask in masks
        ]


def _tensor_payload(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy()


def _tensor_sequence_payload(
    values: Iterable[torch.Tensor],
    *,
    batch_size: int,
) -> dict[str, np.ndarray]:
    return {
        f"layer_{index}": _batch_major_payload(_tensor_payload(value), batch_size)
        for index, value in enumerate(values)
    }


def _batch_major_payload(value: np.ndarray, batch_size: int) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim == 0:
        raise ValueError("Slot payload must include a batch dimension.")
    if array.shape[0] == batch_size:
        return np.ascontiguousarray(array)
    if array.shape[0] % batch_size != 0:
        raise ValueError(
            "Slot payload cannot be flattened batch-major because its first "
            f"dimension {array.shape[0]} is not divisible by batch size {batch_size}."
        )
    return np.ascontiguousarray(array.reshape(batch_size, -1))


__all__ = [
    "PredictEngine",
    "SlotInferenceBatchOutput",
    "SlotOutputPayload",
]
