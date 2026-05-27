import typing as _T

import numpy as np
import torch as _torch

from .checkpoint_engine import CheckpointEngine as _CheckpointEngine
from .fit_engine import FitEngine as _FitEngine
from .locality_runtime_helpers import (
    build_locality_context_if_required as _build_locality_context_if_required,
    slot_locality_kwargs as _slot_locality_kwargs,
)
from .runtime_plugin_bundle_contract import RuntimePluginBundle
from .plugins.dataloader.contracts import DataLoaderPlugin
from .plugins.feature_agg.contracts import FeatureAggregatorPlugin
from .plugins.backbone.contracts import BackbonePlugin
from .plugins.distance.contracts import (
    DistanceAnomalyScorer,
    DistancePlugin,
)
from .plugins.materialize.contracts import MaterializationPlugin
from .plugins.mem_agg.contracts import MemoryAggregationPlugin
from .plugins.patch_align.contracts import PatchAlignPlugin
from .plugins.locality_context_contract import LocalityContext
from ..util.param_binding import normalize_training_contract
from .plugins.preprocess.contracts import PreprocessPlugin
from .plugins.proj1.contracts import Projector1Plugin
from .plugins.proj2.contracts import Projector2Plugin
from .predict_engine import (
    PredictEngine as _PredictEngine,
    SlotInferenceBatchOutput,
)
from .plugins.scoring.contracts import ScoringPlugin, ScoringSegmentor
from .plugins.transform.contracts import TransformPlugin
from .inference_pipeline import InferencePipeline as _InferencePipeline
from .inference_output_contract import InferenceBatchOutput, InferenceDatasetOutput
from .train_pipeline import TrainPipeline as _TrainPipeline
from .plugins.backbone.contracts import BackboneFeatureExtractor
from .pipeline_stage_contract import train_pipeline_stage_order

_DEFAULT_TRAINABLE_STAGES: frozenset[str] = frozenset(
    {"feature_agg", "proj1", "transform", "proj2", "mem_agg"}
)
_EXPLICIT_TRAINABLE_STAGE_ORDER: tuple[str, ...] = tuple(
    stage_name
    for stage_name in train_pipeline_stage_order()
    if stage_name in _DEFAULT_TRAINABLE_STAGES
)


class MHPatchCore(_torch.nn.Module):
    def __init__(
        self,
        *,
        device: _torch.device,
        training_contract: str = "OFFLINE",
        fit_epochs: _T.Mapping[str, int] | None = None,
        plugin_bundle: RuntimePluginBundle | None = None,
    ):
        """Initialize from orchestration metadata and a bound runtime plugin bundle."""
        normalized_training_contract = normalize_training_contract(training_contract)

        fit_epochs_map = self._validate_fit_epochs_map(fit_epochs)
        reduction_passes = self._require_positive_int(
            int(fit_epochs_map["feature_agg"]),
            "pipeline.training.fit_epochs.feature_agg",
        )
        projector1_passes = self._require_positive_int(
            int(fit_epochs_map["proj1"]),
            "pipeline.training.fit_epochs.proj1",
        )
        covariance_passes = self._require_positive_int(
            int(fit_epochs_map["transform"]),
            "pipeline.training.fit_epochs.transform",
        )
        projector2_passes = self._require_positive_int(
            int(fit_epochs_map["proj2"]),
            "pipeline.training.fit_epochs.proj2",
        )
        aggregation_passes = self._require_positive_int(
            int(fit_epochs_map["mem_agg"]),
            "pipeline.training.fit_epochs.mem_agg",
        )

        super().__init__()
        self._device = device
        if plugin_bundle is None:
            raise ValueError(
                "plugin_bundle is required for MHPatchCore construction. "
                "Resolve plugins via canonical startup flow "
                "(config -> discover -> select -> bind -> validate -> compile)."
            )
        resolved_runtime_bundle = plugin_bundle
        resolved_plugin_bundle = resolved_runtime_bundle.model_plugin_bundle
        self._training_contract = normalized_training_contract

        self._dataloader_plugin: DataLoaderPlugin = (
            resolved_runtime_bundle.dataloader_plugin
        )
        self._backbone_plugin: BackbonePlugin = resolved_plugin_bundle.backbone_plugin
        self._patch_align_plugin: PatchAlignPlugin = (
            resolved_plugin_bundle.patch_align_plugin
        )
        self._distance_plugin: DistancePlugin = resolved_plugin_bundle.distance_plugin
        self._preprocess_plugin: PreprocessPlugin = (
            resolved_plugin_bundle.preprocess_plugin
        )
        self._feature_agg_plugin: FeatureAggregatorPlugin = (
            resolved_plugin_bundle.feature_agg_plugin
        )
        self._mem_agg_plugin: MemoryAggregationPlugin = (
            resolved_plugin_bundle.mem_agg_plugin
        )
        self._materialize_plugin: MaterializationPlugin = (
            resolved_plugin_bundle.materialize_plugin
        )
        self._scoring_plugin: ScoringPlugin = resolved_plugin_bundle.scoring_plugin
        self._proj1_plugin: Projector1Plugin = resolved_plugin_bundle.proj1_plugin
        self._transform_plugin: TransformPlugin = (
            resolved_plugin_bundle.transform_plugin
        )
        self._proj2_plugin: Projector2Plugin = (
            resolved_plugin_bundle.proj2_plugin
        )

        stage_bindings: dict[str, object] = {
            "dataloader": self._dataloader_plugin,
            "backbone": self._backbone_plugin,
            "patch_align": self._patch_align_plugin,
            "preprocess": self._preprocess_plugin,
            "feature_agg": self._feature_agg_plugin,
            "proj1": self._proj1_plugin,
            "transform": self._transform_plugin,
            "proj2": self._proj2_plugin,
            "mem_agg": self._mem_agg_plugin,
            "materialize": self._materialize_plugin,
            "distance": self._distance_plugin,
            "scoring": self._scoring_plugin,
        }

        _MISSING = object()

        def _read_mode_capability(
            *,
            plugin: object,
            stage_name: str,
            capability_name: str,
            default_value: object,
        ) -> bool:
            raw_value = getattr(plugin, capability_name, _MISSING)
            if raw_value is _MISSING:
                if default_value is _MISSING:
                    raise TypeError(
                        "Selected plugin is missing required stage mode capability: "
                        f"stage='{stage_name}' capability='{capability_name}' "
                        f"plugin_type='{type(plugin).__name__}'"
                    )
                raw_value = default_value
            if not isinstance(raw_value, bool):
                raise TypeError(
                    "Stage mode capability must be boolean: "
                    f"stage='{stage_name}' capability='{capability_name}' "
                    f"type='{type(raw_value).__name__}'"
                )
            return raw_value

        self._pipeline_stage_mode_capabilities: dict[str, dict[str, bool]] = {}
        for stage_name, plugin in stage_bindings.items():
            requires_explicit_mode_capabilities = stage_name in {
                "distance",
                "scoring",
            }
            self._pipeline_stage_mode_capabilities[stage_name] = {
                "supports_train": _read_mode_capability(
                    plugin=plugin,
                    stage_name=stage_name,
                    capability_name="supports_train",
                    default_value=(
                        _MISSING if requires_explicit_mode_capabilities else True
                    ),
                ),
                "supports_inference": _read_mode_capability(
                    plugin=plugin,
                    stage_name=stage_name,
                    capability_name="supports_inference",
                    default_value=(
                        _MISSING if requires_explicit_mode_capabilities else True
                    ),
                ),
            }

        self._input_shape = self._resolve_input_shape()
        self._backbone, self._resolved_embedding_layers = (
            self._initialize_backbone_and_layers()
        )
        self._anomaly_score_num_nn = int(self._distance_plugin.resolve_num_neighbors())
        if self._anomaly_score_num_nn <= 0:
            raise ValueError(
                "distance plugin returned invalid num_neighbors; "
                f"got {self._anomaly_score_num_nn}."
            )

        self._pipeline_stage_fit_epochs: dict[str, int] = {
            stage_name: 1 for stage_name in train_pipeline_stage_order()
        }
        self._pipeline_stage_fit_epochs["feature_agg"] = int(reduction_passes)
        self._pipeline_stage_fit_epochs["proj1"] = int(projector1_passes)
        self._pipeline_stage_fit_epochs["transform"] = int(covariance_passes)
        self._pipeline_stage_fit_epochs["proj2"] = int(projector2_passes)
        self._pipeline_stage_fit_epochs["mem_agg"] = int(aggregation_passes)
        self._pipeline_stage_trainability: dict[str, bool] = {
            stage_name: stage_name in _DEFAULT_TRAINABLE_STAGES
            for stage_name in train_pipeline_stage_order()
        }

        self.patch_maker = self._patch_align_plugin.create_patch_maker()
        reduction_selection = self._feature_agg_plugin.resolve_reduction_selection(
            training_contract=self._training_contract,
        )
        self._feature_agg_requires_fit_state = bool(
            self._feature_agg_plugin.requires_fit_state(
                selection=reduction_selection,
            )
        )

        self.forward_modules = _torch.nn.ModuleDict({})

        # Preprocessing (Average Pooling)
        self._initialize_feature_aggregator()

        self.anomaly_scorer, self.anomaly_segmentor = self._initialize_distance_backend(
            anomaly_score_num_nn=self._anomaly_score_num_nn,
            input_shape=self._input_shape,
        )

        self._stage_owned_state: dict[str, dict[str, _T.Any]] = {
            "feature_agg": {},
            "proj1": {},
            "transform": {},
            "proj2": {},
            "mem_agg": {},
            "materialize": {},
            "scoring": {},
        }
        self._proj1_plugin.state_load(state=None)
        self._transform_plugin.state_load(state=None)
        self._proj2_plugin.state_load(state=None)
        self._fit_engine = _FitEngine(self)
        self._predict_engine = _PredictEngine(self)
        self._checkpoint_engine = _CheckpointEngine(self)
        self._train_pipeline = _TrainPipeline(self)
        self._inference_pipeline = _InferencePipeline(self)

    @staticmethod
    def _require_positive_int(value: _T.Any, field_name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{field_name} must be a positive integer.")
        return int(value)

    @staticmethod
    def _validate_fit_epochs_map(
        fit_epochs: _T.Mapping[str, int] | None,
    ) -> dict[str, int]:
        if fit_epochs is None:
            raise ValueError(
                "pipeline.training.fit_epochs must define exactly the explicit "
                "trainable stages: feature_agg, proj1, transform, proj2, mem_agg."
            )
        if not isinstance(fit_epochs, _T.Mapping):
            raise TypeError("pipeline.training.fit_epochs must be a mapping.")
        fit_epochs_map = {str(stage_name): stage_value for stage_name, stage_value in fit_epochs.items()}
        expected_keys = set(_EXPLICIT_TRAINABLE_STAGE_ORDER)
        actual_keys = set(fit_epochs_map.keys())
        if actual_keys != expected_keys:
            raise ValueError(
                "pipeline.training.fit_epochs must define exactly the explicit "
                "trainable stages: feature_agg, proj1, transform, proj2, mem_agg."
            )
        return fit_epochs_map

    def set_stage_fit_epochs(self, stage_name: str, fit_epochs: int) -> None:
        """Set TRAIN fit-epoch count for one canonical stage."""

        if stage_name not in self._pipeline_stage_fit_epochs:
            raise KeyError(f"Unknown TRAIN stage '{stage_name}'.")
        if isinstance(fit_epochs, bool) or not isinstance(fit_epochs, int) or fit_epochs <= 0:
            raise ValueError("fit_epochs must be a positive integer.")
        self._pipeline_stage_fit_epochs[stage_name] = int(fit_epochs)

    def set_stage_trainability(self, stage_name: str, trainable: bool) -> None:
        """Set TRAIN trainability flag for one canonical stage."""

        if stage_name not in self._pipeline_stage_trainability:
            raise KeyError(f"Unknown TRAIN stage '{stage_name}'.")
        if not isinstance(trainable, bool):
            raise ValueError("trainable must be a boolean.")
        self._pipeline_stage_trainability[stage_name] = trainable

    def _stage_state_slot(self, stage_name: str) -> dict[str, _T.Any]:
        slot = self._stage_owned_state.get(stage_name)
        if slot is None:
            slot = {}
            self._stage_owned_state[stage_name] = slot
        return slot

    def _uses_transform_state(self) -> bool:
        """Return whether selected transform plugin requires fitted state."""
        return bool(getattr(self._transform_plugin, "requires_fit_state", False))

    def _uses_proj1_state(self) -> bool:
        """Return whether selected proj1 plugin requires fitted state."""
        return bool(getattr(self._proj1_plugin, "requires_fit_state", False))

    def _uses_proj2_state(self) -> bool:
        """Return whether selected proj2 plugin requires fitted state."""
        return bool(getattr(self._proj2_plugin, "requires_fit_state", False))

    def _requires_feature_agg_fit_state(self) -> bool:
        """Return whether selected feature-aggregation setup requires fit state."""
        return bool(getattr(self, "_feature_agg_requires_fit_state", True))

    def _requires_patch_scoring_state(self) -> bool:
        """Return whether selected scoring plugin requires fit-time aux state."""
        scoring_plugin = getattr(self, "_scoring_plugin", None)
        if scoring_plugin is None:
            return False
        return bool(getattr(scoring_plugin, "requires_patch_scoring_state", False))

    def _set_stage_owned_state(self, *, stage_name: str, key: str, value: _T.Any) -> None:
        self._stage_state_slot(stage_name)[key] = value

    def _get_stage_owned_state(
        self,
        *,
        stage_name: str,
        key: str,
        default: _T.Any = None,
    ) -> _T.Any:
        return self._stage_state_slot(stage_name).get(key, default)

    def _resolve_input_shape(self) -> tuple[int, int]:
        """Resolve canonical input shape from the bound dataloader plugin."""
        input_shape = self._dataloader_plugin.resolve_input_shape()
        if not isinstance(input_shape, tuple) or len(input_shape) != 2:
            raise TypeError(
                "dataloader plugin must resolve input_shape as tuple[int, int]; "
                f"got {input_shape!r}."
            )
        return (
            self._require_positive_int(int(input_shape[0]), "input_shape[0]"),
            self._require_positive_int(int(input_shape[1]), "input_shape[1]"),
        )

    def _initialize_backbone_and_layers(self) -> tuple[_torch.nn.Module, list[str]]:
        """Delegate backbone construction and resolved-layer wiring to component."""
        return self._backbone_plugin.initialize_backbone_and_layers(
            device=self._device,
        )

    def _create_feature_extractor(self) -> BackboneFeatureExtractor:
        """Delegate feature-extractor creation to component."""
        return self._backbone_plugin.create_feature_extractor(
            backbone=self._backbone,
            resolved_embedding_layers=self._resolved_embedding_layers,
        )

    def _patchify_and_align_features(
        self,
        features: list[_torch.Tensor],
    ) -> tuple[list[_torch.Tensor], list[list[int]]]:
        """Delegate patchification and spatial alignment to component."""
        return self._patch_align_plugin.patchify_and_align(
            features=features,
            patch_maker=self.patch_maker,
        )

    def _capture_embed_features(
        self,
        images: _torch.Tensor,
    ) -> list[_torch.Tensor]:
        """Capture embedding features with deterministic layer ordering."""
        extractor = self._create_feature_extractor()
        try:
            features = extractor(images)
        finally:
            extractor.close()
        return [features[layer] for layer in self._resolved_embedding_layers]

    def _forward_embed_post_capture(
        self,
        features: list[_torch.Tensor],
        locality_context: LocalityContext | None = None,
    ) -> _torch.Tensor:
        """Delegate post-capture embedding forwards using fixed module keys/order."""
        processed = self._preprocess_plugin.forward_embed_preprocess(
            features=features,
            forward_modules=self.forward_modules,
        )
        processed = self._feature_agg_plugin.forward_embed_feature_aggregation(
            features=processed,
            forward_modules=self.forward_modules,
        )
        processed = self._proj1_plugin.forward_embed_projector1(
            features=processed,
            forward_modules=self.forward_modules,
            **_slot_locality_kwargs(
                plugin=self._proj1_plugin,
                locality_context=locality_context,
            ),
        )
        processed = self._transform_plugin.forward_embed_transform(
            features=processed,
            forward_modules=self.forward_modules,
            **_slot_locality_kwargs(
                plugin=self._transform_plugin,
                locality_context=locality_context,
            ),
        )
        processed = self._proj2_plugin.forward_embed_projector2(
            features=processed,
            forward_modules=self.forward_modules,
            **_slot_locality_kwargs(
                plugin=self._proj2_plugin,
                locality_context=locality_context,
            ),
        )
        return processed

    def _initialize_distance_backend(
        self,
        anomaly_score_num_nn: int,
        input_shape: tuple[int, int],
    ) -> tuple[DistanceAnomalyScorer, ScoringSegmentor]:
        """Build distance backend with current deterministic wiring."""
        anomaly_scorer = self._distance_plugin.create_anomaly_scorer(
            n_nearest_neighbours=anomaly_score_num_nn
        )
        anomaly_segmentor = self._scoring_plugin.create_segmentor(
            device=self._device,
            target_size=input_shape[-2:],
        )
        return anomaly_scorer, anomaly_segmentor

    @staticmethod
    def _validate_finite_feature_array(
        features: np.ndarray,
        stage: str,
        batch_idx: int | None = None,
    ) -> None:
        """Fail fast when non-finite values appear in extracted features."""
        finite_mask = np.isfinite(features)
        if np.all(finite_mask):
            return
        non_finite_count = int(features.size - int(finite_mask.sum()))
        batch_suffix = f", batch={batch_idx}" if batch_idx is not None else ""
        raise ValueError(
            f"Non-finite feature values detected during {stage}{batch_suffix}. "
            f"non_finite_count={non_finite_count}"
        )

    def _initialize_feature_aggregator(self) -> None:
        dummy_input = _torch.zeros((1, 3, *self._input_shape)).to(self._device)
        extractor = self._create_feature_extractor()
        try:
            features = extractor(dummy_input)
        finally:
            extractor.close()

        feature_dims = [
            features[layer].shape[1] for layer in self._resolved_embedding_layers
        ]
        preprocessing = self._preprocess_plugin.create_preprocessing_module(
            input_dims=feature_dims,
        ).to(self._device)
        preadapt_aggregator = (
            self._feature_agg_plugin.create_preadapt_aggregator_module().to(
                self._device
            )
        )

        self.forward_modules["preprocessing"] = preprocessing
        self.forward_modules["preadapt_aggregator"] = preadapt_aggregator

    def _embed(
        self,
        images: _torch.Tensor,
        detach: bool = True,
        provide_patch_shapes: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, list[list[int]]]:
        """
        Returns feature embeddings for images.

        Extracts features from backbone, patches them, and aggregates them.
        """
        def _detach(features: _torch.Tensor) -> np.ndarray:
            if detach:
                return features.detach().cpu().numpy()
            return features.cpu().numpy()

        features = self._capture_embed_features(images)
        features, patch_shapes = self._patchify_and_align_features(features)
        locality_context = _build_locality_context_if_required(
            batch_size=int(images.shape[0]),
            patch_shapes=patch_shapes,
            plugins=(self._proj1_plugin, self._transform_plugin, self._proj2_plugin),
        )

        # Preprocessing & Aggregation
        embedded = self._forward_embed_post_capture(
            features,
            locality_context=locality_context,
        )

        if provide_patch_shapes:
            return _detach(embedded), patch_shapes
        return _detach(embedded)

    def fit(self, data: _T.Iterable[_T.Any]) -> "MHPatchCore":
        return self._train_pipeline.fit(data)

    def infer(
        self,
        data: _torch.utils.data.DataLoader[_T.Any] | _torch.Tensor,
    ) -> InferenceBatchOutput | InferenceDatasetOutput:
        return self._inference_pipeline.infer(data)

    def infer_dataloader(
        self,
        dataloader: _torch.utils.data.DataLoader[_T.Any],
    ) -> InferenceDatasetOutput:
        return self._inference_pipeline.infer_dataloader(dataloader)

    def infer_batch(self, images: _torch.Tensor) -> InferenceBatchOutput:
        return self._inference_pipeline.infer_batch(images)

    def infer_batch_with_slot_outputs(
        self,
        images: _torch.Tensor,
        *,
        selected_slots: _T.Iterable[str],
    ) -> SlotInferenceBatchOutput:
        return self._predict_engine.predict_batch_with_slot_outputs(
            images,
            selected_slots=selected_slots,
        )

    def save_to_path(self, save_path: str, prepend: str = "") -> None:
        self._checkpoint_engine.save_to_path(save_path, prepend)

    def load_from_path(self, load_path: str, prepend: str = "") -> None:
        self._checkpoint_engine.load_from_path(load_path, prepend)
