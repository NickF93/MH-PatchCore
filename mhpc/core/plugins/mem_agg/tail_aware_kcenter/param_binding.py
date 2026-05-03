"""Plugin-local parameter binding for `tail_aware_kcenter`."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from typing import Any

from ..contracts import MemAggBindContextLike

_ALLOWED_KEYS = frozenset(
    {
        "n_clusters",
        "cluster_range",
        "streaming_enforce_cluster_budget",
        "kcenter_chunk_coreset_size",
        "tail_selection_strategy",
        "main_budget_fraction",
        "tail_probability_min",
        "tail_probability_max",
        "geometric_candidate_pool_size",
        "phase1_passes",
        "deduplication_strategy",
        "dedup_quantization_decimals",
        "dedup_norm_tolerance",
    }
)

_TAIL_SELECTION_STRATEGIES = frozenset(
    {
        "chi2_band",
        "geometric_residual",
        "geometric_main_residual",
        "geometric_pruning_gap",
    }
)
_GEOMETRIC_TAIL_SELECTION_STRATEGIES = frozenset(
    {
        "geometric_residual",
        "geometric_main_residual",
        "geometric_pruning_gap",
    }
)
_TAIL_SELECTION_STRATEGY_MESSAGE = (
    "tail_selection_strategy must be one of: "
    "chi2_band, geometric_residual, geometric_main_residual, "
    "geometric_pruning_gap."
)
_DEDUPLICATION_STRATEGIES = frozenset(
    {"exact_row", "quantized_row", "norm_tolerance"}
)
_DEDUPLICATION_STRATEGY_MESSAGE = (
    "deduplication_strategy must be one of: exact_row, quantized_row, "
    "norm_tolerance."
)


@dataclass(frozen=True)
class TailAwareKCenterParams:
    """Plugin-local params for `tail_aware_kcenter`."""

    n_clusters: int | str
    cluster_range: tuple[int, int] | str
    streaming_enforce_cluster_budget: bool
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


def _require_positive_int(*, value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer.")
    return int(value)


def _require_probability(*, value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError(f"{field_name} must be a float in (0, 1).")
    normalized = float(value)
    if not 0.0 < normalized < 1.0:
        raise ValueError(f"{field_name} must be a float in (0, 1).")
    return normalized


def _require_positive_finite_float(*, value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError(f"{field_name} must be a finite float > 0.")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized <= 0.0:
        raise ValueError(f"{field_name} must be a finite float > 0.")
    return normalized


def _ensure_allowed_keys(*, params: Mapping[str, Any], context: str) -> None:
    unknown_keys = sorted(str(key) for key in set(params.keys()) - _ALLOWED_KEYS)
    if unknown_keys:
        raise ValueError(f"{context} contains unsupported keys: {', '.join(unknown_keys)}")


def _parse_mem_agg_params(params: Mapping[str, Any]) -> TailAwareKCenterParams:
    _ensure_allowed_keys(params=params, context="pipeline.slots.mem_agg.params")
    raw_n_clusters = params.get("n_clusters", "auto")
    if isinstance(raw_n_clusters, str):
        if raw_n_clusters != "auto":
            raise ValueError("n_clusters must be 'auto' or a positive integer.")
        n_clusters: int | str = "auto"
    elif isinstance(raw_n_clusters, int) and not isinstance(raw_n_clusters, bool):
        if raw_n_clusters <= 0:
            raise ValueError("n_clusters must be 'auto' or a positive integer.")
        n_clusters = int(raw_n_clusters)
    else:
        raise ValueError("n_clusters must be 'auto' or a positive integer.")

    raw_cluster_range = params.get("cluster_range", "auto")
    cluster_range: tuple[int, int] | str
    if isinstance(raw_cluster_range, str):
        if raw_cluster_range != "auto":
            raise ValueError("cluster_range must be 'auto' or a [min, max] integer pair.")
        cluster_range = "auto"
    elif isinstance(raw_cluster_range, (list, tuple)) and len(raw_cluster_range) == 2:
        cluster_min = _require_positive_int(
            value=raw_cluster_range[0],
            field_name="cluster_range[0]",
        )
        cluster_max = _require_positive_int(
            value=raw_cluster_range[1],
            field_name="cluster_range[1]",
        )
        if cluster_min > cluster_max:
            raise ValueError("cluster_range must satisfy min <= max.")
        cluster_range = (cluster_min, cluster_max)
    else:
        raise ValueError("cluster_range must be 'auto' or a [min, max] integer pair.")

    streaming_enforce_cluster_budget = params.get("streaming_enforce_cluster_budget", True)
    if not isinstance(streaming_enforce_cluster_budget, bool):
        raise ValueError("streaming_enforce_cluster_budget must be a boolean.")

    kcenter_chunk_coreset_size = _require_positive_int(
        value=params.get("kcenter_chunk_coreset_size", 256),
        field_name="kcenter_chunk_coreset_size",
    )
    raw_tail_selection_strategy = params.get("tail_selection_strategy")
    if not isinstance(raw_tail_selection_strategy, str):
        raise ValueError(_TAIL_SELECTION_STRATEGY_MESSAGE)
    tail_selection_strategy = raw_tail_selection_strategy.strip().lower()
    if tail_selection_strategy not in _TAIL_SELECTION_STRATEGIES:
        raise ValueError(_TAIL_SELECTION_STRATEGY_MESSAGE)

    main_budget_fraction = _require_probability(
        value=params.get("main_budget_fraction", 0.85),
        field_name="main_budget_fraction",
    )

    has_tail_probability_min = "tail_probability_min" in params
    has_tail_probability_max = "tail_probability_max" in params
    has_geometric_candidate_pool_size = "geometric_candidate_pool_size" in params
    tail_probability_min: float | None
    tail_probability_max: float | None
    geometric_candidate_pool_size: int | None
    if tail_selection_strategy == "chi2_band":
        if has_geometric_candidate_pool_size:
            raise ValueError(
                "geometric_candidate_pool_size is only supported when "
                "tail_selection_strategy is geometric_residual or "
                "geometric_main_residual or geometric_pruning_gap."
            )
        if not has_tail_probability_min or not has_tail_probability_max:
            raise ValueError(
                "chi2_band tail selection requires tail_probability_min and "
                "tail_probability_max."
            )
        tail_probability_min = _require_probability(
            value=params["tail_probability_min"],
            field_name="tail_probability_min",
        )
        tail_probability_max = _require_probability(
            value=params["tail_probability_max"],
            field_name="tail_probability_max",
        )
        if tail_probability_min >= tail_probability_max:
            raise ValueError(
                "tail_probability_min must be < tail_probability_max for "
                "tail_aware_kcenter."
            )
        geometric_candidate_pool_size = None
    elif tail_selection_strategy in _GEOMETRIC_TAIL_SELECTION_STRATEGIES:
        if has_tail_probability_min or has_tail_probability_max:
            raise ValueError(
                "tail_probability_min and tail_probability_max are only supported "
                "when tail_selection_strategy is chi2_band."
            )
        if not has_geometric_candidate_pool_size:
            raise ValueError(
                "geometric tail selection requires "
                "geometric_candidate_pool_size."
            )
        tail_probability_min = None
        tail_probability_max = None
        geometric_candidate_pool_size = _require_positive_int(
            value=params["geometric_candidate_pool_size"],
            field_name="geometric_candidate_pool_size",
        )
    else:
        raise ValueError(_TAIL_SELECTION_STRATEGY_MESSAGE)

    phase1_passes = _require_positive_int(
        value=params.get("phase1_passes", 1),
        field_name="phase1_passes",
    )
    raw_deduplication_strategy = params.get("deduplication_strategy")
    if not isinstance(raw_deduplication_strategy, str):
        raise ValueError(_DEDUPLICATION_STRATEGY_MESSAGE)
    deduplication_strategy = raw_deduplication_strategy.strip().lower()
    if deduplication_strategy not in _DEDUPLICATION_STRATEGIES:
        raise ValueError(_DEDUPLICATION_STRATEGY_MESSAGE)

    has_dedup_quantization_decimals = "dedup_quantization_decimals" in params
    has_dedup_norm_tolerance = "dedup_norm_tolerance" in params
    dedup_quantization_decimals: int | None
    dedup_norm_tolerance: float | None
    if deduplication_strategy == "exact_row":
        if has_dedup_quantization_decimals or has_dedup_norm_tolerance:
            raise ValueError(
                "dedup_quantization_decimals and dedup_norm_tolerance are not "
                "supported when deduplication_strategy is exact_row."
            )
        dedup_quantization_decimals = None
        dedup_norm_tolerance = None
    elif deduplication_strategy == "quantized_row":
        if not has_dedup_quantization_decimals:
            raise ValueError(
                "quantized_row deduplication requires "
                "dedup_quantization_decimals."
            )
        if has_dedup_norm_tolerance:
            raise ValueError(
                "dedup_norm_tolerance is only supported when "
                "deduplication_strategy is norm_tolerance."
            )
        dedup_quantization_decimals = _require_positive_int(
            value=params["dedup_quantization_decimals"],
            field_name="dedup_quantization_decimals",
        )
        dedup_norm_tolerance = None
    elif deduplication_strategy == "norm_tolerance":
        if has_dedup_quantization_decimals:
            raise ValueError(
                "dedup_quantization_decimals is only supported when "
                "deduplication_strategy is quantized_row."
            )
        if not has_dedup_norm_tolerance:
            raise ValueError(
                "norm_tolerance deduplication requires dedup_norm_tolerance."
            )
        dedup_quantization_decimals = None
        dedup_norm_tolerance = _require_positive_finite_float(
            value=params["dedup_norm_tolerance"],
            field_name="dedup_norm_tolerance",
        )
    else:
        raise ValueError(_DEDUPLICATION_STRATEGY_MESSAGE)

    return TailAwareKCenterParams(
        n_clusters=n_clusters,
        cluster_range=cluster_range,
        streaming_enforce_cluster_budget=streaming_enforce_cluster_budget,
        kcenter_chunk_coreset_size=kcenter_chunk_coreset_size,
        tail_selection_strategy=tail_selection_strategy,
        main_budget_fraction=main_budget_fraction,
        tail_probability_min=tail_probability_min,
        tail_probability_max=tail_probability_max,
        geometric_candidate_pool_size=geometric_candidate_pool_size,
        phase1_passes=phase1_passes,
        deduplication_strategy=deduplication_strategy,
        dedup_quantization_decimals=dedup_quantization_decimals,
        dedup_norm_tolerance=dedup_norm_tolerance,
    )


class TailAwareKCenterParamBindingMixin:
    """Plugin-local parameter binding for `tail_aware_kcenter`."""

    _bound_params: dict[str, Any]
    _bound_bind_context: MemAggBindContextLike
    _mem_agg_params: TailAwareKCenterParams

    def bind_params(
        self,
        *,
        params: Mapping[str, Any],
        bind_context: MemAggBindContextLike,
    ) -> None:
        if not isinstance(params, Mapping):
            raise TypeError(
                "params must be a mapping for plugin bind_params: "
                f"type={type(params).__name__}"
            )
        training_contract = getattr(bind_context, "training_contract", None)
        if not isinstance(training_contract, str):
            raise TypeError(
                "bind_context.training_contract must be a string: "
                f"type={type(training_contract).__name__}"
            )
        normalized_contract = training_contract.strip().upper()
        if normalized_contract != "STREAMING":
            raise ValueError(
                "tail_aware_kcenter aggregation is supported only in STREAMING mode."
            )
        seed = getattr(bind_context, "seed", None)
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError(
                "bind_context.seed must be an integer: "
                f"type={type(seed).__name__}"
            )
        self._bound_params = dict(params)
        self._bound_bind_context = bind_context
        self._mem_agg_params = _parse_mem_agg_params(self._bound_params)

    def _resolve_n_clusters(self, *, feature_count: int | None) -> int | None:
        params = self._mem_agg_params
        if isinstance(params.n_clusters, int):
            return int(params.n_clusters)
        cluster_range = params.cluster_range
        if isinstance(cluster_range, str):
            min_c, max_c = 1, 1000
        else:
            min_c, max_c = cluster_range
        if feature_count is None:
            return int(max_c)
        estimated = int(feature_count * 0.1)
        return max(min(estimated, int(max_c)), int(min_c))
