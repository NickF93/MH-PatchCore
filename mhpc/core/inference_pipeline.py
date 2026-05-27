"""Generic INFERENCE pipeline coordinator with typed output contracts."""

from __future__ import annotations

from typing import Any

import torch

from .batching import normalize_batch
from .inference_output_contract import (
    InferenceBatchOutput,
    InferenceDatasetOutput,
    as_inference_batch_output,
    as_inference_dataset_output,
)
from .pipeline_stage_contract import (
    assert_stage_allowed_in_mode,
    inference_pipeline_stage_order,
)


class InferencePipeline:
    """Host-owned INFERENCE coordinator using canonical stage contracts."""

    def __init__(self, model: Any) -> None:
        self._model = model

    def resolve_stage_order(self) -> tuple[str, ...]:
        """Return canonical INFERENCE stage order."""

        return inference_pipeline_stage_order()

    def _validate_inference_contracts(self) -> None:
        for stage_name in self.resolve_stage_order():
            assert_stage_allowed_in_mode(
                stage_name,
                execution_mode="inference",
                stage_mode_capabilities=getattr(
                    self._model,
                    "_pipeline_stage_mode_capabilities",
                    None,
                ),
            )

    def infer_batch(self, images: torch.Tensor) -> InferenceBatchOutput:
        """Run one-batch inference and return typed output."""

        self._validate_inference_contracts()
        raw_output = self._model._predict_engine.predict_batch(images)
        return as_inference_batch_output(raw_output)

    def infer_dataloader(
        self,
        dataloader: torch.utils.data.DataLoader[Any],
    ) -> InferenceDatasetOutput:
        """Run dataloader inference and return typed output."""

        self._validate_inference_contracts()
        raw_output = self._model._predict_engine.predict_dataloader(dataloader)
        return as_inference_dataset_output(raw_output)

    def infer(
        self,
        data: torch.utils.data.DataLoader[Any] | torch.Tensor,
    ) -> InferenceBatchOutput | InferenceDatasetOutput:
        """Run inference over tensor or dataloader and return typed output."""

        self._validate_inference_contracts()
        raw_output = self._model._predict_engine.predict(data)
        if isinstance(data, torch.utils.data.DataLoader):
            return as_inference_dataset_output(raw_output)
        normalized = normalize_batch(data, include_targets=False)
        if not isinstance(normalized.images, torch.Tensor):
            raise TypeError("Normalized batch images must be a tensor.")
        return as_inference_batch_output(raw_output)
