"""Coverage-first tail-aware k-center runtime."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
from scipy.spatial import distance as scipy_distance  # type: ignore[import-untyped]
from scipy.stats import chi2 as scipy_chi2  # type: ignore[import-untyped]

from ..contracts import AggregationRuntimeMetadata, TrainUpdateContext

_TAIL_STRATEGY_CHI2_BAND = "chi2_band"
_TAIL_STRATEGY_GEOMETRIC_RESIDUAL = "geometric_residual"
_TAIL_STRATEGY_GEOMETRIC_MAIN_RESIDUAL = "geometric_main_residual"
_TAIL_STRATEGY_GEOMETRIC_PRUNING_GAP = "geometric_pruning_gap"
_GEOMETRIC_TAIL_STRATEGIES = frozenset(
    {
        _TAIL_STRATEGY_GEOMETRIC_RESIDUAL,
        _TAIL_STRATEGY_GEOMETRIC_MAIN_RESIDUAL,
        _TAIL_STRATEGY_GEOMETRIC_PRUNING_GAP,
    }
)
_MAIN_PROVISIONAL_REFERENCE_STRATEGIES = frozenset(
    {_TAIL_STRATEGY_CHI2_BAND, _TAIL_STRATEGY_GEOMETRIC_MAIN_RESIDUAL}
)
_TAIL_SELECTION_STRATEGY_MESSAGE = (
    "tail_selection_strategy must be one of: "
    "chi2_band, geometric_residual, geometric_main_residual, "
    "geometric_pruning_gap."
)
_DEDUP_STRATEGY_EXACT_ROW = "exact_row"
_DEDUP_STRATEGY_QUANTIZED_ROW = "quantized_row"
_DEDUP_STRATEGY_NORM_TOLERANCE = "norm_tolerance"
_DEDUPLICATION_STRATEGIES = frozenset(
    {
        _DEDUP_STRATEGY_EXACT_ROW,
        _DEDUP_STRATEGY_QUANTIZED_ROW,
        _DEDUP_STRATEGY_NORM_TOLERANCE,
    }
)
_DEDUPLICATION_STRATEGY_MESSAGE = (
    "deduplication_strategy must be one of: exact_row, quantized_row, "
    "norm_tolerance."
)


def _farthest_first_coreset(data: np.ndarray, target_size: int) -> np.ndarray:
    """Select a deterministic farthest-first coreset from a 2D feature array."""
    data_np = np.asarray(data, dtype=np.float64)
    if data_np.ndim != 2:
        raise ValueError(
            f"Expected 2D feature matrix for coreset selection, got {data_np.shape}"
        )
    if int(data_np.shape[0]) == 0:
        raise ValueError("Cannot build a coreset from an empty feature matrix.")
    if target_size <= 0:
        raise ValueError("target_size must be > 0 for coreset selection.")

    n_samples = int(data_np.shape[0])
    k = min(int(target_size), n_samples)
    if k == n_samples:
        return np.ascontiguousarray(data_np.copy(), dtype=np.float64)

    mean_vector = np.mean(data_np, axis=0, dtype=np.float64, keepdims=True)
    dist_to_mean = np.linalg.norm(data_np - mean_vector, axis=1)
    first_idx = int(np.argmax(dist_to_mean))

    selected_indices: list[int] = [first_idx]
    min_distances = np.linalg.norm(data_np - data_np[first_idx], axis=1)
    min_distances[first_idx] = 0.0

    while len(selected_indices) < k:
        next_idx = int(np.argmax(min_distances))
        selected_indices.append(next_idx)
        new_distances = np.linalg.norm(data_np - data_np[next_idx], axis=1)
        min_distances = np.minimum(min_distances, new_distances)
        min_distances[next_idx] = 0.0

    return np.ascontiguousarray(
        data_np[np.asarray(selected_indices, dtype=np.int64)],
        dtype=np.float64,
    )


def _stable_top_k_indices(*, scores: np.ndarray, top_k: int) -> np.ndarray:
    scores_np = np.asarray(scores, dtype=np.float64)
    if scores_np.ndim != 1:
        raise ValueError(f"Expected 1D scores, got {scores_np.shape}.")
    if top_k <= 0:
        raise ValueError("top_k must be > 0.")
    if int(scores_np.shape[0]) <= int(top_k):
        return np.arange(int(scores_np.shape[0]), dtype=np.int64)
    order = np.argsort(-scores_np, kind="stable")
    return np.asarray(order[:top_k], dtype=np.int64)


def _min_distances_to_pool(*, points: np.ndarray, pool: np.ndarray) -> np.ndarray:
    points_np = np.asarray(points, dtype=np.float64)
    pool_np = np.asarray(pool, dtype=np.float64)
    if int(pool_np.shape[0]) == 0:
        raise ValueError("Distance-to-pool scoring requires a non-empty pool.")
    distances = np.asarray(
        scipy_distance.cdist(points_np, pool_np, metric="euclidean"),
        dtype=np.float64,
    )
    return np.min(distances, axis=1).astype(np.float64, copy=False)


@dataclass(frozen=True)
class _DeduplicationConfig:
    strategy: str
    quantization_decimals: int | None
    norm_tolerance: float | None


def _resolve_deduplication_config(
    *,
    deduplication_strategy: str,
    dedup_quantization_decimals: int | None,
    dedup_norm_tolerance: float | None,
) -> _DeduplicationConfig:
    normalized_strategy = str(deduplication_strategy).strip().lower()
    if normalized_strategy not in _DEDUPLICATION_STRATEGIES:
        raise ValueError(_DEDUPLICATION_STRATEGY_MESSAGE)
    if normalized_strategy == _DEDUP_STRATEGY_EXACT_ROW:
        if dedup_quantization_decimals is not None or dedup_norm_tolerance is not None:
            raise ValueError(
                "dedup_quantization_decimals and dedup_norm_tolerance are not "
                "supported when deduplication_strategy is exact_row."
            )
        return _DeduplicationConfig(
            strategy=normalized_strategy,
            quantization_decimals=None,
            norm_tolerance=None,
        )
    if normalized_strategy == _DEDUP_STRATEGY_QUANTIZED_ROW:
        if isinstance(dedup_quantization_decimals, bool) or not isinstance(
            dedup_quantization_decimals,
            int,
        ):
            raise ValueError("dedup_quantization_decimals must be a positive integer.")
        if dedup_quantization_decimals <= 0:
            raise ValueError("dedup_quantization_decimals must be a positive integer.")
        if dedup_norm_tolerance is not None:
            raise ValueError(
                "dedup_norm_tolerance is only supported when "
                "deduplication_strategy is norm_tolerance."
            )
        return _DeduplicationConfig(
            strategy=normalized_strategy,
            quantization_decimals=int(dedup_quantization_decimals),
            norm_tolerance=None,
        )
    if dedup_quantization_decimals is not None:
        raise ValueError(
            "dedup_quantization_decimals is only supported when "
            "deduplication_strategy is quantized_row."
        )
    if isinstance(dedup_norm_tolerance, bool) or not isinstance(
        dedup_norm_tolerance,
        (float, int),
    ):
        raise ValueError("dedup_norm_tolerance must be a finite float > 0.")
    normalized_tolerance = float(dedup_norm_tolerance)
    if not math.isfinite(normalized_tolerance) or normalized_tolerance <= 0.0:
        raise ValueError("dedup_norm_tolerance must be a finite float > 0.")
    return _DeduplicationConfig(
        strategy=normalized_strategy,
        quantization_decimals=None,
        norm_tolerance=normalized_tolerance,
    )


def _row_exists(*, row: np.ndarray, pool: np.ndarray) -> bool:
    if int(pool.shape[0]) == 0:
        return False
    return bool(np.any(np.all(pool == row[None, :], axis=1)))


def _stable_unique_rows(points: np.ndarray) -> np.ndarray:
    points_np = np.asarray(points, dtype=np.float64)
    if points_np.ndim != 2:
        raise ValueError(f"Expected 2D points, got {points_np.shape}.")
    if int(points_np.shape[0]) <= 1:
        return np.ascontiguousarray(points_np, dtype=np.float64)

    seen: set[tuple[float, ...]] = set()
    keep_indices: list[int] = []
    for row_index, row in enumerate(points_np):
        row_key = tuple(float(value) for value in row)
        if row_key in seen:
            continue
        seen.add(row_key)
        keep_indices.append(row_index)

    return np.ascontiguousarray(
        points_np[np.asarray(keep_indices, dtype=np.int64)],
        dtype=np.float64,
    )


def _stable_unique_rows_with_scores(
    *,
    points: np.ndarray,
    scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    points_np = np.asarray(points, dtype=np.float64)
    scores_np = np.asarray(scores, dtype=np.float64)
    if points_np.ndim != 2:
        raise ValueError(f"Expected 2D points, got {points_np.shape}.")
    if scores_np.ndim != 1:
        raise ValueError(f"Expected 1D scores, got {scores_np.shape}.")
    if int(points_np.shape[0]) != int(scores_np.shape[0]):
        raise ValueError("points and scores must contain the same number of rows.")
    if int(points_np.shape[0]) <= 1:
        return (
            np.ascontiguousarray(points_np, dtype=np.float64),
            np.ascontiguousarray(scores_np, dtype=np.float64),
        )

    seen: set[tuple[float, ...]] = set()
    keep_indices: list[int] = []
    for row_index, row in enumerate(points_np):
        row_key = tuple(float(value) for value in row)
        if row_key in seen:
            continue
        seen.add(row_key)
        keep_indices.append(row_index)

    index_array = np.asarray(keep_indices, dtype=np.int64)
    return (
        np.ascontiguousarray(points_np[index_array], dtype=np.float64),
        np.ascontiguousarray(scores_np[index_array], dtype=np.float64),
    )


def _quantized_row_key(*, row: np.ndarray, decimals: int) -> tuple[float, ...]:
    rounded = np.round(np.asarray(row, dtype=np.float64), decimals=int(decimals))
    return tuple(float(value) for value in rounded)


def _stable_unique_row_indices(
    *,
    points: np.ndarray,
    deduplication_config: _DeduplicationConfig,
) -> np.ndarray:
    points_np = np.asarray(points, dtype=np.float64)
    if points_np.ndim != 2:
        raise ValueError(f"Expected 2D points, got {points_np.shape}.")
    if int(points_np.shape[0]) <= 1:
        return np.arange(int(points_np.shape[0]), dtype=np.int64)

    if deduplication_config.strategy == _DEDUP_STRATEGY_EXACT_ROW:
        seen: set[tuple[float, ...]] = set()
        exact_keep_indices: list[int] = []
        for row_index, row in enumerate(points_np):
            row_key = tuple(float(value) for value in row)
            if row_key in seen:
                continue
            seen.add(row_key)
            exact_keep_indices.append(row_index)
        return np.asarray(exact_keep_indices, dtype=np.int64)

    if deduplication_config.strategy == _DEDUP_STRATEGY_QUANTIZED_ROW:
        decimals = deduplication_config.quantization_decimals
        if decimals is None:
            raise RuntimeError("quantized_row deduplication requires decimals.")
        seen_quantized: set[tuple[float, ...]] = set()
        quantized_keep_indices: list[int] = []
        for row_index, row in enumerate(points_np):
            row_key = _quantized_row_key(row=row, decimals=decimals)
            if row_key in seen_quantized:
                continue
            seen_quantized.add(row_key)
            quantized_keep_indices.append(row_index)
        return np.asarray(quantized_keep_indices, dtype=np.int64)

    tolerance = deduplication_config.norm_tolerance
    if tolerance is None:
        raise RuntimeError("norm_tolerance deduplication requires tolerance.")
    tolerance_keep_indices: list[int] = []
    kept_rows: list[np.ndarray] = []
    for row_index, row in enumerate(points_np):
        if any(
            float(np.linalg.norm(kept_row - row)) <= float(tolerance)
            for kept_row in kept_rows
        ):
            continue
        tolerance_keep_indices.append(row_index)
        kept_rows.append(row.copy())
    return np.asarray(tolerance_keep_indices, dtype=np.int64)


def _deduplicate_rows(
    *,
    points: np.ndarray,
    deduplication_config: _DeduplicationConfig,
) -> np.ndarray:
    points_np = np.asarray(points, dtype=np.float64)
    keep_indices = _stable_unique_row_indices(
        points=points_np,
        deduplication_config=deduplication_config,
    )
    return np.ascontiguousarray(points_np[keep_indices], dtype=np.float64)


def _deduplicate_rows_with_scores(
    *,
    points: np.ndarray,
    scores: np.ndarray,
    deduplication_config: _DeduplicationConfig,
) -> tuple[np.ndarray, np.ndarray]:
    points_np = np.asarray(points, dtype=np.float64)
    scores_np = np.asarray(scores, dtype=np.float64)
    if scores_np.ndim != 1:
        raise ValueError(f"Expected 1D scores, got {scores_np.shape}.")
    if int(points_np.shape[0]) != int(scores_np.shape[0]):
        raise ValueError("points and scores must contain the same number of rows.")
    keep_indices = _stable_unique_row_indices(
        points=points_np,
        deduplication_config=deduplication_config,
    )
    return (
        np.ascontiguousarray(points_np[keep_indices], dtype=np.float64),
        np.ascontiguousarray(scores_np[keep_indices], dtype=np.float64),
    )


def _row_exists_with_deduplication(
    *,
    row: np.ndarray,
    pool: np.ndarray,
    deduplication_config: _DeduplicationConfig,
) -> bool:
    pool_np = np.asarray(pool, dtype=np.float64)
    if int(pool_np.shape[0]) == 0:
        return False
    row_np = np.asarray(row, dtype=np.float64)
    if deduplication_config.strategy == _DEDUP_STRATEGY_EXACT_ROW:
        return _row_exists(row=row_np, pool=pool_np)
    if deduplication_config.strategy == _DEDUP_STRATEGY_QUANTIZED_ROW:
        decimals = deduplication_config.quantization_decimals
        if decimals is None:
            raise RuntimeError("quantized_row deduplication requires decimals.")
        row_key = _quantized_row_key(row=row_np, decimals=decimals)
        return any(
            _quantized_row_key(row=pool_row, decimals=decimals) == row_key
            for pool_row in pool_np
        )
    tolerance = deduplication_config.norm_tolerance
    if tolerance is None:
        raise RuntimeError("norm_tolerance deduplication requires tolerance.")
    distances = np.linalg.norm(pool_np - row_np, axis=1)
    return bool(np.any(distances <= float(tolerance)))


def _filter_existing_rows(
    *,
    points: np.ndarray,
    pool: np.ndarray,
    deduplication_config: _DeduplicationConfig,
) -> np.ndarray:
    points_np = np.asarray(points, dtype=np.float64)
    pool_np = np.asarray(pool, dtype=np.float64)
    if int(points_np.shape[0]) == 0 or int(pool_np.shape[0]) == 0:
        return _deduplicate_rows(
            points=points_np,
            deduplication_config=deduplication_config,
        )
    keep_mask = np.asarray(
        [
            not _row_exists_with_deduplication(
                row=row,
                pool=pool_np,
                deduplication_config=deduplication_config,
            )
            for row in points_np
        ],
        dtype=np.bool_,
    )
    return _deduplicate_rows(
        points=points_np[keep_mask],
        deduplication_config=deduplication_config,
    )


def _select_farthest_candidates_from_pool(
    *,
    candidates: np.ndarray,
    initial_pool: np.ndarray,
    target_size: int,
    deduplication_config: _DeduplicationConfig,
) -> np.ndarray:
    candidates_np = _deduplicate_rows(
        points=candidates,
        deduplication_config=deduplication_config,
    )
    initial_pool_np = np.asarray(initial_pool, dtype=np.float64)
    if target_size <= 0:
        raise ValueError("target_size must be > 0.")
    if int(candidates_np.shape[0]) == 0:
        feature_dim = (
            int(initial_pool_np.shape[1])
            if initial_pool_np.ndim == 2 and int(initial_pool_np.shape[0]) > 0
            else 0
        )
        return np.empty((0, feature_dim), dtype=np.float64)
    if int(initial_pool_np.shape[0]) == 0:
        return _farthest_first_coreset(candidates_np, target_size=target_size)

    selected_indices: list[int] = []
    selected_mask = np.zeros(int(candidates_np.shape[0]), dtype=np.bool_)
    min_distances = _min_distances_to_pool(
        points=candidates_np,
        pool=initial_pool_np,
    )
    selection_count = min(int(target_size), int(candidates_np.shape[0]))
    while len(selected_indices) < selection_count:
        masked_distances = np.where(selected_mask, -np.inf, min_distances)
        next_idx = int(np.argmax(masked_distances))
        selected_indices.append(next_idx)
        selected_mask[next_idx] = True
        new_distances = np.linalg.norm(
            candidates_np - candidates_np[next_idx],
            axis=1,
        )
        min_distances = np.minimum(min_distances, new_distances)

    return np.ascontiguousarray(
        candidates_np[np.asarray(selected_indices, dtype=np.int64)],
        dtype=np.float64,
    )


@dataclass(frozen=True)
class TailAwareKCenterRuntimeConfig:
    """Resolved runtime config for `tail_aware_kcenter`."""

    n_clusters: int
    main_budget: int
    tail_budget: int
    kcenter_chunk_coreset_size: int
    tail_selection_strategy: str
    main_budget_fraction: float
    tail_probability_min: float | None
    tail_probability_max: float | None
    geometric_candidate_pool_size: int | None
    phase1_passes: int
    deduplication_strategy: str
    dedup_quantization_decimals: int | None
    dedup_norm_tolerance: float | None


class _MergeReduceKCenterPool:
    """Plugin-local merge-reduce k-center coreset pool."""

    def __init__(self, *, target_size: int, chunk_coreset_size: int) -> None:
        if target_size <= 0:
            raise ValueError("target_size must be > 0.")
        if chunk_coreset_size <= 0:
            raise ValueError("chunk_coreset_size must be > 0.")
        self._target_size = int(target_size)
        self._chunk_coreset_size = int(chunk_coreset_size)
        self._seen_samples = 0
        self._levels: dict[int, list[np.ndarray]] = {}

    def update(self, batch: np.ndarray) -> None:
        batch_np = np.asarray(batch, dtype=np.float64)
        if batch_np.ndim != 2:
            raise ValueError(f"Expected 2D feature batch, got {batch_np.shape}.")
        if int(batch_np.shape[0]) == 0:
            return

        local_size = min(
            self._target_size,
            self._chunk_coreset_size,
            int(batch_np.shape[0]),
        )
        local_block = _farthest_first_coreset(batch_np, target_size=local_size)
        self._push_merge_reduce_block(local_block)
        self._seen_samples += int(batch_np.shape[0])

    def centers(self, *, target_size: int | None = None) -> np.ndarray:
        if self._seen_samples <= 0:
            return np.empty((0, 0), dtype=np.float64)

        all_blocks: list[np.ndarray] = []
        for level in sorted(self._levels):
            all_blocks.extend(self._levels[level])
        if not all_blocks:
            return np.empty((0, 0), dtype=np.float64)

        resolved_target_size = (
            self._target_size if target_size is None else int(target_size)
        )
        if resolved_target_size <= 0:
            raise ValueError("target_size must be > 0.")
        resolved_target_size = min(resolved_target_size, self._target_size)
        merged = np.concatenate(all_blocks, axis=0)
        final_size = min(resolved_target_size, int(merged.shape[0]))
        return _farthest_first_coreset(merged, target_size=final_size)

    def _push_merge_reduce_block(self, block: np.ndarray) -> None:
        candidate = np.ascontiguousarray(block, dtype=np.float64)
        level = 0
        while True:
            slots = self._levels.setdefault(level, [])
            slots.append(candidate)
            if len(slots) < 2:
                return

            left = slots.pop(0)
            right = slots.pop(0)
            merged = np.concatenate([left, right], axis=0)
            reduced_size = min(
                self._target_size,
                self._chunk_coreset_size,
                int(merged.shape[0]),
            )
            candidate = _farthest_first_coreset(
                merged,
                target_size=reduced_size,
            )
            level += 1


class TailAwareKCenterAggregationStrategy:
    """Coverage-first k-center runtime with bounded tail-aware capacity."""

    def __init__(
        self,
        *,
        n_clusters: int,
        chunk_coreset_size: int,
        tail_selection_strategy: str,
        main_budget_fraction: float,
        tail_probability_min: float | None,
        tail_probability_max: float | None,
        geometric_candidate_pool_size: int | None,
        phase1_passes: int,
        deduplication_strategy: str,
        dedup_quantization_decimals: int | None,
        dedup_norm_tolerance: float | None,
        enforce_reference_limit: bool = True,
    ) -> None:
        if n_clusters <= 1:
            raise ValueError("TailAwareKCenterAggregationStrategy requires n_clusters > 1.")
        if not 0.0 < main_budget_fraction < 1.0:
            raise ValueError("main_budget_fraction must be in (0, 1).")
        normalized_tail_selection_strategy = str(tail_selection_strategy).strip().lower()
        uses_main_provisional_reference = (
            normalized_tail_selection_strategy
            in _MAIN_PROVISIONAL_REFERENCE_STRATEGIES
        )
        if normalized_tail_selection_strategy not in {
            _TAIL_STRATEGY_CHI2_BAND,
            _TAIL_STRATEGY_GEOMETRIC_RESIDUAL,
            _TAIL_STRATEGY_GEOMETRIC_MAIN_RESIDUAL,
            _TAIL_STRATEGY_GEOMETRIC_PRUNING_GAP,
        }:
            raise ValueError(_TAIL_SELECTION_STRATEGY_MESSAGE)
        if normalized_tail_selection_strategy == _TAIL_STRATEGY_CHI2_BAND:
            if tail_probability_min is None or tail_probability_max is None:
                raise ValueError(
                    "chi2_band tail selection requires tail_probability_min and "
                    "tail_probability_max."
                )
            if not 0.0 < tail_probability_min < tail_probability_max < 1.0:
                raise ValueError(
                    "tail_probability_min and tail_probability_max must satisfy "
                    "0 < min < max < 1."
                )
            if geometric_candidate_pool_size is not None:
                raise ValueError(
                    "geometric_candidate_pool_size is only supported when "
                    "tail_selection_strategy is geometric_residual or "
                    "geometric_main_residual or geometric_pruning_gap."
                )
        else:
            if tail_probability_min is not None or tail_probability_max is not None:
                raise ValueError(
                    "tail_probability_min and tail_probability_max are only supported "
                    "when tail_selection_strategy is chi2_band."
                )
            if geometric_candidate_pool_size is None or geometric_candidate_pool_size <= 0:
                raise ValueError(
                    "geometric_candidate_pool_size must be > 0 for "
                    "geometric tail selection."
                )
        if phase1_passes <= 0:
            raise ValueError("phase1_passes must be > 0.")
        if chunk_coreset_size <= 0:
            raise ValueError("chunk_coreset_size must be > 0.")
        deduplication_config = _resolve_deduplication_config(
            deduplication_strategy=deduplication_strategy,
            dedup_quantization_decimals=dedup_quantization_decimals,
            dedup_norm_tolerance=dedup_norm_tolerance,
        )

        main_budget = int(round(float(main_budget_fraction) * int(n_clusters)))
        main_budget = max(1, min(int(n_clusters) - 1, main_budget))
        tail_budget = int(n_clusters) - main_budget
        if normalized_tail_selection_strategy in _GEOMETRIC_TAIL_STRATEGIES:
            if geometric_candidate_pool_size is None:
                raise RuntimeError("geometric strategy requires candidate pool.")
            if int(geometric_candidate_pool_size) < tail_budget:
                raise ValueError(
                    "geometric_candidate_pool_size must be >= computed tail_budget "
                    "for geometric tail selection."
                )
        self._config = TailAwareKCenterRuntimeConfig(
            n_clusters=int(n_clusters),
            main_budget=main_budget,
            tail_budget=tail_budget,
            kcenter_chunk_coreset_size=int(chunk_coreset_size),
            tail_selection_strategy=normalized_tail_selection_strategy,
            main_budget_fraction=float(main_budget_fraction),
            tail_probability_min=(
                None if tail_probability_min is None else float(tail_probability_min)
            ),
            tail_probability_max=(
                None if tail_probability_max is None else float(tail_probability_max)
            ),
            geometric_candidate_pool_size=(
                None
                if geometric_candidate_pool_size is None
                else int(geometric_candidate_pool_size)
            ),
            phase1_passes=int(phase1_passes),
            deduplication_strategy=deduplication_config.strategy,
            dedup_quantization_decimals=deduplication_config.quantization_decimals,
            dedup_norm_tolerance=deduplication_config.norm_tolerance,
        )
        self._deduplication_config = deduplication_config
        self._runtime_metadata = AggregationRuntimeMetadata(
            reference_limit=int(n_clusters),
            enforce_reference_limit=bool(enforce_reference_limit),
        )
        self._observed_epoch_count: int | None = None
        self._last_epoch_index: int | None = None
        self._dimension: int | None = None
        self._seen_samples = 0
        self._seen_updates = 0
        self._phase1_updates = 0
        self._phase2_updates = 0
        self._phase2_started = False
        self._tail_seen = 0
        self._tail_retained_candidates = 0
        self._coverage_pool = _MergeReduceKCenterPool(
            target_size=int(n_clusters),
            chunk_coreset_size=int(chunk_coreset_size),
        )
        # If both budgets exceed the chunk cap, merge-reduce block reductions
        # are identical, so sharing preserves math while avoiding duplicate work.
        self._main_coverage_pool = (
            self._coverage_pool
            if main_budget >= int(chunk_coreset_size)
            else _MergeReduceKCenterPool(
                target_size=main_budget,
                chunk_coreset_size=int(chunk_coreset_size),
            )
        )
        self._provisional_pool = (
            _MergeReduceKCenterPool(
                target_size=int(n_clusters),
                chunk_coreset_size=int(chunk_coreset_size),
            )
            if normalized_tail_selection_strategy
            in {
                _TAIL_STRATEGY_GEOMETRIC_RESIDUAL,
                _TAIL_STRATEGY_GEOMETRIC_PRUNING_GAP,
            }
            else None
        )
        self._provisional_centers: np.ndarray | None = None
        self._pruning_gap_main_provisional_centers: np.ndarray | None = None
        self._main_provisional_pool = (
            _MergeReduceKCenterPool(
                target_size=main_budget,
                chunk_coreset_size=int(chunk_coreset_size),
            )
            if uses_main_provisional_reference
            else None
        )
        self._main_provisional_centers: np.ndarray | None = None
        self._tail_candidate_pool: np.ndarray | None = None
        self._tail_candidate_scores: np.ndarray | None = None

    @property
    def name(self) -> str:
        return "TAIL_AWARE_KCENTER_STREAMING"

    @property
    def is_streaming(self) -> bool:
        return True

    @property
    def supports_multi_pass(self) -> bool:
        return True

    def update(
        self,
        batch: np.ndarray,
        locality_context: object | None = None,
        update_context: object | None = None,
    ) -> None:
        del locality_context
        context = self._validate_update_context(update_context=update_context)
        batch_np = np.asarray(batch, dtype=np.float64)
        if batch_np.ndim != 2:
            raise ValueError(
                f"Expected 2D feature batch [N,D], got shape={batch_np.shape}"
            )
        if batch_np.shape[0] == 0:
            return
        if not np.all(np.isfinite(batch_np)):
            raise ValueError(
                "TailAwareKCenterAggregationStrategy received non-finite features."
            )
        self._validate_or_set_dimension(feature_dim=int(batch_np.shape[1]))
        self._last_epoch_index = int(context.epoch_index)
        self._seen_samples += int(batch_np.shape[0])
        self._seen_updates += 1
        if context.epoch_index <= self._config.phase1_passes:
            self._phase1_updates += 1
            self._update_phase1(batch_np)
            return

        self._update_phase2(batch_np)

    def get_centroids(self) -> np.ndarray:
        if self._seen_updates <= 0:
            raise RuntimeError("TailAwareKCenterAggregationStrategy received no batches.")
        main_centers = self._main_coverage_pool.centers(
            target_size=self._config.main_budget
        )
        if int(main_centers.shape[0]) == 0:
            raise RuntimeError(
                "TailAwareKCenterAggregationStrategy produced no protected centers."
            )

        selected_parts = [main_centers]
        if self._config.tail_budget > 0 and self._tail_candidate_pool is not None:
            tail_candidates = _filter_existing_rows(
                points=self._tail_candidate_pool,
                pool=main_centers,
                deduplication_config=self._deduplication_config,
            )
            if int(tail_candidates.shape[0]) > 0:
                selected_parts.append(
                    self._select_final_tail_representatives(
                        tail_candidates=tail_candidates,
                        main_centers=main_centers,
                    )
                )

        combined = np.ascontiguousarray(np.concatenate(selected_parts, axis=0))
        if int(combined.shape[0]) < self._config.n_clusters:
            fallback_candidates = _filter_existing_rows(
                points=self._coverage_pool.centers(
                    target_size=self._config.n_clusters
                ),
                pool=combined,
                deduplication_config=self._deduplication_config,
            )
            if int(fallback_candidates.shape[0]) > 0:
                fill_count = min(
                    self._config.n_clusters - int(combined.shape[0]),
                    int(fallback_candidates.shape[0]),
                )
                selected_fill = _select_farthest_candidates_from_pool(
                    candidates=fallback_candidates,
                    initial_pool=combined,
                    target_size=fill_count,
                    deduplication_config=self._deduplication_config,
                )
                combined = np.ascontiguousarray(
                    np.concatenate([combined, selected_fill], axis=0),
                    dtype=np.float64,
                )

        if int(combined.shape[0]) > self._config.n_clusters:
            combined = combined[: self._config.n_clusters]
        return np.ascontiguousarray(combined, dtype=np.float64)

    def export_state(self) -> dict[str, Any]:
        main_centers = self._main_coverage_pool.centers(
            target_size=self._config.main_budget
        )
        fallback_centers = self._coverage_pool.centers(
            target_size=self._config.n_clusters
        )
        tail_candidate_count = (
            0
            if self._tail_candidate_pool is None
            else int(self._tail_candidate_pool.shape[0])
        )
        provisional_center_count = self._provisional_center_count()
        stream_state: dict[str, Any] = {
            "type": "tail_aware_kcenter",
            "tail_selection_strategy": self._config.tail_selection_strategy,
            "deduplication_strategy": self._config.deduplication_strategy,
            "main_budget_fraction": float(self._config.main_budget_fraction),
            "main_budget": int(self._config.main_budget),
            "tail_budget": int(self._config.tail_budget),
            "kcenter_chunk_coreset_size": int(self._config.kcenter_chunk_coreset_size),
            "phase1_passes": int(self._config.phase1_passes),
            "phase1_updates": int(self._phase1_updates),
            "phase2_updates": int(self._phase2_updates),
            "seen_samples": int(self._seen_samples),
            "seen_updates": int(self._seen_updates),
            "epoch_count": self._observed_epoch_count,
            "dimension": self._dimension,
            "main_pool_size": int(main_centers.shape[0]),
            "fallback_pool_size": int(fallback_centers.shape[0]),
            "tail_candidate_pool_size": tail_candidate_count,
            "tail_seen": int(self._tail_seen),
            "tail_retained_candidates": int(self._tail_retained_candidates),
        }
        if self._config.tail_selection_strategy == _TAIL_STRATEGY_CHI2_BAND:
            stream_state["tail_probability_min"] = self._config.tail_probability_min
            stream_state["tail_probability_max"] = self._config.tail_probability_max
            stream_state["provisional_reference_target"] = "main_budget"
            stream_state["provisional_center_count"] = (
                self._main_provisional_center_count()
            )
        elif (
            self._config.tail_selection_strategy
            == _TAIL_STRATEGY_GEOMETRIC_MAIN_RESIDUAL
        ):
            stream_state["geometric_candidate_pool_size"] = (
                self._config.geometric_candidate_pool_size
            )
            stream_state["provisional_reference_target"] = "main_budget"
            stream_state["provisional_center_count"] = (
                self._main_provisional_center_count()
            )
        elif (
            self._config.tail_selection_strategy
            == _TAIL_STRATEGY_GEOMETRIC_PRUNING_GAP
        ):
            stream_state["geometric_candidate_pool_size"] = (
                self._config.geometric_candidate_pool_size
            )
            stream_state["provisional_reference_target"] = "pruning_gap"
            stream_state["full_provisional_center_count"] = provisional_center_count
            stream_state["main_provisional_center_count"] = (
                self._pruning_gap_main_provisional_center_count()
            )
        else:
            stream_state["geometric_candidate_pool_size"] = (
                self._config.geometric_candidate_pool_size
            )
            stream_state["provisional_reference_target"] = "n_clusters"
            stream_state["provisional_center_count"] = provisional_center_count
        if self._config.deduplication_strategy == _DEDUP_STRATEGY_QUANTIZED_ROW:
            stream_state["dedup_quantization_decimals"] = (
                self._config.dedup_quantization_decimals
            )
        elif self._config.deduplication_strategy == _DEDUP_STRATEGY_NORM_TOLERANCE:
            stream_state["dedup_norm_tolerance"] = self._config.dedup_norm_tolerance
        return {
            "kmeans_model": None,
            "stream_state": stream_state,
        }

    def runtime_metadata(self) -> AggregationRuntimeMetadata:
        return self._runtime_metadata

    def _validate_update_context(
        self,
        *,
        update_context: object | None,
    ) -> TrainUpdateContext:
        if not isinstance(update_context, TrainUpdateContext):
            raise TypeError(
                "TailAwareKCenterAggregationStrategy requires TrainUpdateContext updates."
            )
        if update_context.epoch_count < 2:
            raise ValueError("tail_aware_kcenter requires fit_epochs.mem_agg >= 2.")
        if self._config.phase1_passes >= update_context.epoch_count:
            raise ValueError(
                "phase1_passes must be < fit_epochs.mem_agg for tail_aware_kcenter."
            )
        if update_context.epoch_index <= 0 or update_context.batch_index <= 0:
            raise ValueError(
                "TrainUpdateContext must carry positive epoch_index and batch_index."
            )
        if update_context.epoch_index > update_context.epoch_count:
            raise ValueError("TrainUpdateContext epoch_index must be <= epoch_count.")
        if self._observed_epoch_count is None:
            self._observed_epoch_count = int(update_context.epoch_count)
        elif int(update_context.epoch_count) != int(self._observed_epoch_count):
            raise ValueError(
                "TailAwareKCenterAggregationStrategy received inconsistent epoch_count values."
            )
        if (
            self._last_epoch_index is not None
            and int(update_context.epoch_index) < int(self._last_epoch_index)
        ):
            raise ValueError(
                "TailAwareKCenterAggregationStrategy requires monotonic "
                "non-decreasing epoch_index values."
            )
        return update_context

    def _validate_or_set_dimension(self, *, feature_dim: int) -> None:
        if self._dimension is None:
            self._dimension = int(feature_dim)
            return
        if int(feature_dim) != int(self._dimension):
            raise ValueError(
                "TailAwareKCenterAggregationStrategy received inconsistent "
                f"feature dimensions: expected {self._dimension}, got {feature_dim}."
            )

    def _update_phase1(self, batch: np.ndarray) -> None:
        if self._phase2_started:
            raise ValueError(
                "TailAwareKCenterAggregationStrategy cannot accept phase-1 "
                "updates after phase 2 has started."
            )
        if (
            self._config.tail_selection_strategy
            in _MAIN_PROVISIONAL_REFERENCE_STRATEGIES
        ):
            if self._main_provisional_pool is None:
                raise RuntimeError(
                    "main-budget provisional strategy requires provisional main pool."
                )
            self._main_provisional_centers = None
            self._main_provisional_pool.update(batch)
            return
        if self._provisional_pool is None:
            raise RuntimeError("Geometric residual strategy requires provisional pool.")
        self._provisional_centers = None
        self._pruning_gap_main_provisional_centers = None
        self._provisional_pool.update(batch)

    def _update_phase2(self, batch: np.ndarray) -> None:
        batch_np = np.asarray(batch, dtype=np.float64)
        self._phase2_started = True
        self._update_coverage_pools(batch_np)
        tail_candidates, tail_scores = self._select_tail_candidates(batch_np)
        if int(tail_candidates.shape[0]) > 0:
            self._tail_seen += int(tail_candidates.shape[0])
            self._append_tail_candidates(tail_candidates, tail_scores=tail_scores)
        self._phase2_updates += 1

    def _update_coverage_pools(self, batch: np.ndarray) -> None:
        self._coverage_pool.update(batch)
        if self._main_coverage_pool is self._coverage_pool:
            return
        self._main_coverage_pool.update(batch)

    def _select_tail_candidates(
        self,
        batch: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        if self._config.tail_selection_strategy == _TAIL_STRATEGY_CHI2_BAND:
            return self._select_chi2_tail_band(batch), None
        if (
            self._config.tail_selection_strategy
            == _TAIL_STRATEGY_GEOMETRIC_PRUNING_GAP
        ):
            return self._select_geometric_pruning_gap_candidates(batch)
        return self._select_geometric_residual_candidates(batch)

    def _select_chi2_tail_band(self, batch: np.ndarray) -> np.ndarray:
        if self._dimension is None:
            raise RuntimeError("Feature dimension must be known before tail scoring.")
        if (
            self._config.tail_probability_min is None
            or self._config.tail_probability_max is None
        ):
            raise RuntimeError("chi2_band strategy requires configured probabilities.")
        batch_np = np.asarray(batch, dtype=np.float64)
        squared_norms = np.sum(np.square(batch_np), axis=1, dtype=np.float64)
        tail_probability = np.asarray(
            scipy_chi2.sf(squared_norms, df=int(self._dimension)),
            dtype=np.float64,
        )
        mask = np.logical_and(
            tail_probability >= self._config.tail_probability_min,
            tail_probability <= self._config.tail_probability_max,
        )
        return np.ascontiguousarray(batch_np[mask], dtype=np.float64)

    def _select_geometric_residual_candidates(
        self,
        batch: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        candidate_pool_size = self._config.geometric_candidate_pool_size
        if candidate_pool_size is None:
            raise RuntimeError(
                "geometric tail strategy requires geometric_candidate_pool_size."
            )
        batch_np = np.asarray(batch, dtype=np.float64)
        provisional_centers = self._geometric_residual_reference_centers()
        residual_scores = _min_distances_to_pool(
            points=batch_np,
            pool=provisional_centers,
        )
        top_count = min(int(candidate_pool_size), int(batch_np.shape[0]))
        top_indices = _stable_top_k_indices(scores=residual_scores, top_k=top_count)
        return (
            np.ascontiguousarray(batch_np[top_indices], dtype=np.float64),
            np.ascontiguousarray(residual_scores[top_indices], dtype=np.float64),
        )

    def _select_geometric_pruning_gap_candidates(
        self,
        batch: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        candidate_pool_size = self._config.geometric_candidate_pool_size
        if candidate_pool_size is None:
            raise RuntimeError(
                "geometric tail strategy requires geometric_candidate_pool_size."
            )
        batch_np = np.asarray(batch, dtype=np.float64)
        raw_gap_scores = self._geometric_pruning_gap_scores(batch_np)
        gap_scores = np.maximum(raw_gap_scores, 0.0).astype(np.float64, copy=False)
        top_count = min(int(candidate_pool_size), int(batch_np.shape[0]))
        top_indices = _stable_top_k_indices(scores=gap_scores, top_k=top_count)
        return (
            np.ascontiguousarray(batch_np[top_indices], dtype=np.float64),
            np.ascontiguousarray(gap_scores[top_indices], dtype=np.float64),
        )

    def _geometric_pruning_gap_scores(self, batch: np.ndarray) -> np.ndarray:
        batch_np = np.asarray(batch, dtype=np.float64)
        full_centers = self._ensure_provisional_centers()
        main_centers = self._ensure_pruning_gap_main_provisional_centers()
        full_residuals = _min_distances_to_pool(
            points=batch_np,
            pool=full_centers,
        )
        main_residuals = _min_distances_to_pool(
            points=batch_np,
            pool=main_centers,
        )
        return np.ascontiguousarray(main_residuals - full_residuals, dtype=np.float64)

    def _append_tail_candidates(
        self,
        tail_candidates: np.ndarray,
        *,
        tail_scores: np.ndarray | None,
    ) -> None:
        if self._config.tail_selection_strategy in _GEOMETRIC_TAIL_STRATEGIES:
            if tail_scores is None:
                raise RuntimeError(
                    "geometric tail candidates require residual scores."
                )
            self._append_geometric_tail_candidates(
                tail_candidates,
                tail_scores=tail_scores,
            )
            return

        self._append_chi2_tail_candidates(tail_candidates)

    def _append_chi2_tail_candidates(self, tail_candidates: np.ndarray) -> None:
        candidates_np = _deduplicate_rows(
            points=tail_candidates,
            deduplication_config=self._deduplication_config,
        )
        if self._tail_candidate_pool is None:
            candidate_pool = candidates_np
        else:
            candidate_pool = np.concatenate(
                [self._tail_candidate_pool, candidates_np],
                axis=0,
            )
        reference_centers = self._ensure_main_provisional_centers()
        self._tail_candidate_pool = self._reduce_tail_candidate_pool(
            np.ascontiguousarray(candidate_pool, dtype=np.float64),
            reference_centers=reference_centers,
        )
        self._tail_retained_candidates = (
            0
            if self._tail_candidate_pool is None
            else int(self._tail_candidate_pool.shape[0])
        )

    def _append_geometric_tail_candidates(
        self,
        tail_candidates: np.ndarray,
        *,
        tail_scores: np.ndarray,
    ) -> None:
        candidates_np, scores_np = _deduplicate_rows_with_scores(
            points=tail_candidates,
            scores=tail_scores,
            deduplication_config=self._deduplication_config,
        )
        if self._tail_candidate_pool is None:
            candidate_pool = candidates_np
            candidate_scores = scores_np
        else:
            if self._tail_candidate_scores is None:
                raise RuntimeError("Geometric candidate pool is missing residual scores.")
            candidate_pool = np.concatenate(
                [self._tail_candidate_pool, candidates_np],
                axis=0,
            )
            candidate_scores = np.concatenate(
                [self._tail_candidate_scores, scores_np],
                axis=0,
            )
        self._tail_candidate_pool, self._tail_candidate_scores = (
            self._reduce_geometric_tail_candidate_pool(
                points=np.ascontiguousarray(candidate_pool, dtype=np.float64),
                scores=np.ascontiguousarray(candidate_scores, dtype=np.float64),
            )
        )
        self._tail_retained_candidates = int(self._tail_candidate_pool.shape[0])

    def _reduce_tail_candidate_pool(
        self,
        candidate_pool: np.ndarray,
        *,
        reference_centers: np.ndarray,
    ) -> np.ndarray:
        candidate_pool_np = _deduplicate_rows(
            points=candidate_pool,
            deduplication_config=self._deduplication_config,
        )
        if int(candidate_pool_np.shape[0]) <= self._config.n_clusters:
            return np.ascontiguousarray(candidate_pool_np, dtype=np.float64)
        reference_np = np.asarray(reference_centers, dtype=np.float64)
        if int(reference_np.shape[0]) == 0:
            raise RuntimeError(
                "chi2_band candidate retention requires non-empty pass-1 "
                "main-budget provisional coverage."
            )
        return _select_farthest_candidates_from_pool(
            candidates=candidate_pool_np,
            initial_pool=reference_np,
            target_size=self._config.n_clusters,
            deduplication_config=self._deduplication_config,
        )

    def _reduce_geometric_tail_candidate_pool(
        self,
        *,
        points: np.ndarray,
        scores: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        candidate_pool_size = self._config.geometric_candidate_pool_size
        if candidate_pool_size is None:
            raise RuntimeError(
                "geometric tail strategy requires geometric_candidate_pool_size."
            )
        unique_points, unique_scores = _deduplicate_rows_with_scores(
            points=points,
            scores=scores,
            deduplication_config=self._deduplication_config,
        )
        if int(unique_points.shape[0]) <= int(candidate_pool_size):
            return unique_points, unique_scores
        keep_indices = _stable_top_k_indices(
            scores=unique_scores,
            top_k=int(candidate_pool_size),
        )
        return (
            np.ascontiguousarray(unique_points[keep_indices], dtype=np.float64),
            np.ascontiguousarray(unique_scores[keep_indices], dtype=np.float64),
        )

    def _select_final_tail_representatives(
        self,
        *,
        tail_candidates: np.ndarray,
        main_centers: np.ndarray,
    ) -> np.ndarray:
        tail_count = min(self._config.tail_budget, int(tail_candidates.shape[0]))
        return _select_farthest_candidates_from_pool(
            candidates=tail_candidates,
            initial_pool=main_centers,
            target_size=tail_count,
            deduplication_config=self._deduplication_config,
        )

    def _ensure_provisional_centers(self) -> np.ndarray:
        if self._provisional_centers is not None:
            return self._provisional_centers
        if self._provisional_pool is None:
            raise RuntimeError("Geometric residual strategy requires provisional pool.")
        provisional_centers = self._provisional_pool.centers(
            target_size=self._config.n_clusters,
        )
        if int(provisional_centers.shape[0]) == 0:
            raise RuntimeError(
                "geometric tail selection requires non-empty pass-1 provisional "
                "coverage."
            )
        self._provisional_centers = np.ascontiguousarray(
            provisional_centers,
            dtype=np.float64,
        )
        return self._provisional_centers

    def _ensure_main_provisional_centers(self) -> np.ndarray:
        if self._main_provisional_centers is not None:
            return self._main_provisional_centers
        if self._main_provisional_pool is None:
            raise RuntimeError(
                "main-budget provisional strategy requires provisional main pool."
            )
        provisional_centers = self._main_provisional_pool.centers(
            target_size=self._config.main_budget,
        )
        if int(provisional_centers.shape[0]) == 0:
            raise RuntimeError(
                "chi2_band candidate retention requires non-empty pass-1 "
                "main-budget provisional coverage."
            )
        self._main_provisional_centers = np.ascontiguousarray(
            provisional_centers,
            dtype=np.float64,
        )
        return self._main_provisional_centers

    def _ensure_pruning_gap_main_provisional_centers(self) -> np.ndarray:
        if self._pruning_gap_main_provisional_centers is not None:
            return self._pruning_gap_main_provisional_centers
        full_centers = self._ensure_provisional_centers()
        target_size = min(self._config.main_budget, int(full_centers.shape[0]))
        if target_size <= 0:
            raise RuntimeError(
                "geometric_pruning_gap tail selection requires non-empty pass-1 "
                "provisional coverage."
            )
        self._pruning_gap_main_provisional_centers = _farthest_first_coreset(
            full_centers,
            target_size=target_size,
        )
        return self._pruning_gap_main_provisional_centers

    def _geometric_residual_reference_centers(self) -> np.ndarray:
        if (
            self._config.tail_selection_strategy
            == _TAIL_STRATEGY_GEOMETRIC_MAIN_RESIDUAL
        ):
            return self._ensure_main_provisional_centers()
        return self._ensure_provisional_centers()

    def _provisional_center_count(self) -> int:
        if self._provisional_centers is not None:
            return int(self._provisional_centers.shape[0])
        if self._provisional_pool is None:
            return 0
        return int(
            self._provisional_pool.centers(target_size=self._config.n_clusters).shape[0]
        )

    def _main_provisional_center_count(self) -> int:
        if self._main_provisional_centers is not None:
            return int(self._main_provisional_centers.shape[0])
        if self._main_provisional_pool is None:
            return 0
        return int(
            self._main_provisional_pool.centers(
                target_size=self._config.main_budget
            ).shape[0]
        )

    def _pruning_gap_main_provisional_center_count(self) -> int:
        if self._pruning_gap_main_provisional_centers is not None:
            return int(self._pruning_gap_main_provisional_centers.shape[0])
        if self._provisional_pool is None:
            return 0
        full_count = self._provisional_center_count()
        if full_count <= 0:
            return 0
        return min(self._config.main_budget, full_count)
