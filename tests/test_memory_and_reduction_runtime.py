import numpy as np

from mhpc.core.plugins.feature_agg.ipca.strategy import (
    StreamingIncrementalPCAReductionStrategy,
)
from mhpc.core.plugins.mem_agg.kcenter.strategy import (
    StreamingKCenterAggregationStrategy,
)
from mhpc.core.plugins.mem_agg.kmeans.strategy import (
    StreamingMiniBatchKMeansAggregationStrategy,
)
from mhpc.core.plugins.mem_agg.tail_aware_kcenter.strategy import (
    TailAwareKCenterAggregationStrategy,
)
from mhpc.core.train_update_context_contract import TrainUpdateContext


def _feature_grid(rows: int = 12, cols: int = 4) -> np.ndarray:
    values = np.arange(rows * cols, dtype=np.float64).reshape(rows, cols)
    return values / np.max(values)


def test_streaming_ipca_is_deterministic_after_finalize() -> None:
    features = _feature_grid(rows=10, cols=4)

    first = StreamingIncrementalPCAReductionStrategy(variance_ratio=0.95)
    second = StreamingIncrementalPCAReductionStrategy(variance_ratio=0.95)
    for start in (0, 5):
        first.update(features[start : start + 5])
        second.update(features[start : start + 5])

    first.finalize()
    second.finalize()

    first_transformed = first.transform(features)
    second_transformed = second.transform(features)
    assert first_transformed.shape == second_transformed.shape
    assert first.output_dimension == second.output_dimension
    np.testing.assert_allclose(first_transformed, second_transformed)


def test_streaming_kcenter_respects_cluster_budget() -> None:
    features = _feature_grid(rows=9, cols=3)
    strategy = StreamingKCenterAggregationStrategy(
        n_clusters=3,
        mode="merge_reduce",
        chunk_coreset_size=2,
    )
    strategy.update(features[:4])
    strategy.update(features[4:])

    centers = strategy.get_centroids()
    assert centers.shape[0] <= 3
    assert centers.shape[1] == 3
    assert strategy.runtime_metadata().reference_limit == 3


def test_streaming_kmeans_respects_cluster_budget() -> None:
    features = _feature_grid(rows=12, cols=3)
    strategy = StreamingMiniBatchKMeansAggregationStrategy(
        n_clusters=3,
        minibatch_size=4,
        random_state=7,
    )
    strategy.update(features[:6])
    strategy.update(features[6:])

    centers = strategy.get_centroids()
    assert centers.shape == (3, 3)
    assert np.isfinite(centers).all()


def test_tail_aware_kcenter_two_phase_geores_respects_budget() -> None:
    features = _feature_grid(rows=16, cols=4)
    strategy = TailAwareKCenterAggregationStrategy(
        n_clusters=4,
        chunk_coreset_size=3,
        tail_selection_strategy="geometric_residual",
        main_budget_fraction=0.75,
        tail_probability_min=None,
        tail_probability_max=None,
        geometric_candidate_pool_size=4,
        phase1_passes=1,
        deduplication_strategy="exact_row",
        dedup_quantization_decimals=None,
        dedup_norm_tolerance=None,
    )
    strategy.update(
        features,
        update_context=TrainUpdateContext(epoch_index=1, epoch_count=2, batch_index=1),
    )
    strategy.update(
        features,
        update_context=TrainUpdateContext(epoch_index=2, epoch_count=2, batch_index=1),
    )

    centers = strategy.get_centroids()
    assert centers.shape == (4, 4)
    assert strategy.runtime_metadata().reference_limit == 4
