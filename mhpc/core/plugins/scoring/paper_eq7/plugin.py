"""`paper_eq7` scoring plugin implementation."""

from __future__ import annotations

import numpy as np

from ..contracts import DistanceQueryPayload, PatchMakerLike, ScoringPlugin
from .param_binding import ScoringParamBindingMixin
from .runtime import ScoringRuntimeMixin, score_paper_eq7


class PaperEq7ScoringPlugin(
    ScoringParamBindingMixin,
    ScoringRuntimeMixin,
    ScoringPlugin,
):
    """Scoring plugin that uses PatchCore paper Eq.7 image scoring."""

    supports_streaming: bool = True
    requires_full_dataset: bool = False
    requires_locality_context: bool = False
    preserves_locality: bool = False
    requires_patch_scoring_state: bool = False

    def score(
        self,
        *,
        features: np.ndarray,
        patch_scores: np.ndarray,
        query_distances: np.ndarray,
        query_nns: np.ndarray,
        distance_query: DistanceQueryPayload | None = None,
        patch_shape: tuple[int, int],
        batchsize: int,
        patch_maker: PatchMakerLike,
        anomaly_scorer: object,
        patch_scoring_mode: str,
        patch_scoring_state: object | None,
        paper_reweight_num_nn: int,
        locality_context: object | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        del locality_context
        return score_paper_eq7(
            features=features,
            patch_scores=patch_scores,
            query_distances=query_distances,
            query_nns=query_nns,
            distance_query=distance_query,
            patch_shape=patch_shape,
            batchsize=batchsize,
            patch_maker=patch_maker,
            anomaly_scorer=anomaly_scorer,
            patch_scoring_mode=patch_scoring_mode,
            patch_scoring_state=patch_scoring_state,
            paper_reweight_num_nn=paper_reweight_num_nn,
            topk_k=self.resolve_scoring_controls().pni_topk_k,
            topk_temperature=self.resolve_scoring_controls().pni_topk_temperature,
            topp_p=self.resolve_scoring_controls().pni_topp_p,
            topp_max_k=self.resolve_scoring_controls().pni_topp_max_k,
        )
