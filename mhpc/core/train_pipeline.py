"""Generic TRAIN pipeline orchestrator with canonical stage-order contracts."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from .pipeline_stage_contract import (
    assert_stage_allowed_in_mode,
    train_pipeline_stage_order,
)
from .stage_lifecycle_contract import StageLifecycleSelection


class TrainPipeline:
    """Host-owned TRAIN orchestrator using canonical stage-order contracts."""

    def __init__(self, model: Any) -> None:
        self._model = model

    def resolve_stage_selections(self) -> tuple[StageLifecycleSelection, ...]:
        """Build lifecycle metadata for canonical TRAIN stage execution."""

        selections: list[StageLifecycleSelection] = []
        for stage_name in train_pipeline_stage_order():
            if stage_name not in self._model._pipeline_stage_fit_epochs:
                raise KeyError(
                    "pipeline stage fit-epoch map is missing canonical stage "
                    f"'{stage_name}'."
                )
            if stage_name not in self._model._pipeline_stage_trainability:
                raise KeyError(
                    "pipeline stage trainability map is missing canonical stage "
                    f"'{stage_name}'."
                )
            fit_epochs = int(self._model._pipeline_stage_fit_epochs[stage_name])
            trainable = bool(self._model._pipeline_stage_trainability[stage_name])
            selections.append(
                StageLifecycleSelection(
                    stage_name=stage_name,
                    role="capability_driven",
                    trainable=trainable,
                    fit_epochs=fit_epochs,
                )
            )
        return tuple(selections)

    @staticmethod
    def _validate_stage_selection(selection: StageLifecycleSelection) -> None:
        if selection.fit_epochs <= 0:
            raise ValueError(
                "pipeline.training.fit_epochs must be positive: "
                f"stage='{selection.stage_name}' fit_epochs={selection.fit_epochs}."
            )
        if not selection.trainable and selection.fit_epochs != 1:
            raise ValueError(
                "Non-trainable stage must keep fit_epochs=1: "
                f"stage='{selection.stage_name}' fit_epochs={selection.fit_epochs}."
            )

    def _validate_train_contracts(
        self,
        selections: tuple[StageLifecycleSelection, ...],
    ) -> None:
        for selection in selections:
            assert_stage_allowed_in_mode(
                selection.stage_name,
                execution_mode="train",
                stage_mode_capabilities=getattr(
                    self._model,
                    "_pipeline_stage_mode_capabilities",
                    None,
                ),
            )
            self._validate_stage_selection(selection)

    def fit(self, data: Iterable[Any]) -> Any:
        """Run TRAIN orchestration contracts, then delegate fit execution."""

        selections = self.resolve_stage_selections()
        self._validate_train_contracts(selections)
        return self._model._fit_engine.fit(data)
