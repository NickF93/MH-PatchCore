"""Contract for model-side plugin bundle wiring."""

from __future__ import annotations

from typing import Protocol

from .plugins.backbone.contracts import BackbonePlugin
from .plugins.feature_agg.contracts import FeatureAggregatorPlugin
from .plugins.distance.contracts import DistancePlugin
from .plugins.materialize.contracts import MaterializationPlugin
from .plugins.mem_agg.contracts import MemoryAggregationPlugin
from .plugins.patch_align.contracts import PatchAlignPlugin
from .plugins.preprocess.contracts import PreprocessPlugin
from .plugins.proj1.contracts import Projector1Plugin
from .plugins.proj2.contracts import Projector2Plugin
from .plugins.scoring.contracts import ScoringPlugin
from .plugins.transform.contracts import TransformPlugin


class ModelPluginBundle(Protocol):
    """Contract-only protocol for model pipeline plugin composition."""

    @property
    def backbone_plugin(self) -> BackbonePlugin: ...

    @property
    def patch_align_plugin(self) -> PatchAlignPlugin: ...

    @property
    def preprocess_plugin(self) -> PreprocessPlugin: ...

    @property
    def feature_agg_plugin(self) -> FeatureAggregatorPlugin: ...

    @property
    def proj1_plugin(self) -> Projector1Plugin: ...

    @property
    def transform_plugin(self) -> TransformPlugin: ...

    @property
    def proj2_plugin(self) -> Projector2Plugin: ...

    @property
    def mem_agg_plugin(self) -> MemoryAggregationPlugin: ...

    @property
    def materialize_plugin(self) -> MaterializationPlugin: ...

    @property
    def distance_plugin(self) -> DistancePlugin: ...

    @property
    def scoring_plugin(self) -> ScoringPlugin: ...
