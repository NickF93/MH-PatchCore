import numpy as np

from mhpc.core.plugins.patch_align.pc_patchify_align.patch_maker import PatchMaker
from mhpc.core.plugins.scoring.paper_eq7.runtime import (
    compute_paper_eq7_image_scores,
)
from mhpc.core.plugins.scoring.reference_max.runtime import (
    compute_reference_image_scores,
)


class _NearestNeighbor:
    def run(
        self,
        n_nearest_neighbours: int,
        query_features: np.ndarray,
        index_features: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        assert index_features is not None
        distances = np.sum(
            (query_features[:, None, :] - index_features[None, :, :]) ** 2,
            axis=2,
        )
        order = np.argsort(distances, axis=1)[:, :n_nearest_neighbours]
        sorted_distances = np.take_along_axis(distances, order, axis=1)
        return sorted_distances, order


class _Scorer:
    def __init__(self, detection_features: np.ndarray) -> None:
        self.detection_features = detection_features
        self.nn_method = _NearestNeighbor()


def test_reference_max_reduces_patch_scores_per_image() -> None:
    patch_maker = PatchMaker(patchsize=3, stride=1)
    patch_scores = np.array(
        [0.1, 0.2, 0.4, 0.3, 1.0, 2.0, 0.5, 0.25],
        dtype=np.float32,
    )

    scores = compute_reference_image_scores(
        patch_maker=patch_maker,
        patch_scores=patch_scores,
        batchsize=2,
    )

    np.testing.assert_allclose(scores, np.array([0.4, 2.0]), rtol=1.0e-6)


def test_paper_eq7_with_single_reweight_neighbor_matches_patch_max() -> None:
    patch_maker = PatchMaker(patchsize=3, stride=1)
    features = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, 1.0],
            [3.0, 1.0],
        ],
        dtype=np.float32,
    )
    query_distances = np.array(
        [[0.1], [0.2], [0.4], [0.3], [1.0], [2.0], [0.5], [0.25]],
        dtype=np.float32,
    )
    query_nns = np.zeros_like(query_distances, dtype=np.int64)
    scorer = _Scorer(detection_features=features.copy())

    scores = compute_paper_eq7_image_scores(
        patch_maker=patch_maker,
        anomaly_scorer=scorer,
        paper_reweight_num_nn=1,
        features=features,
        query_distances=query_distances,
        query_nns=query_nns,
        batchsize=2,
    )

    np.testing.assert_allclose(scores, np.array([0.4, 2.0]), rtol=1.0e-6)
