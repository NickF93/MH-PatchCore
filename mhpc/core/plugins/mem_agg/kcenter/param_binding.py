"""Plugin-local parameter binding/parsing for mem_agg plugins."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ..contracts import MemAggBindContextLike

_ALLOWED_KEYS = frozenset(
    {
        "n_clusters",
        "cluster_range",
        "streaming_enforce_cluster_budget",
        "kcenter_mode",
        "kcenter_chunk_coreset_size",
        "kcenter_distance_threshold",
    }
)


@dataclass(frozen=True)
class MemAggParams:
    """Plugin-local params for `kcenter`."""

    n_clusters: int | str
    cluster_range: tuple[int, int] | str
    streaming_enforce_cluster_budget: bool
    kcenter_mode: str
    kcenter_chunk_coreset_size: int
    kcenter_distance_threshold: float | None


def _require_positive_int(*, value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer.")
    return int(value)


def _ensure_allowed_keys(*, params: Mapping[str, Any], context: str) -> None:
    unknown_keys = sorted(str(key) for key in set(params.keys()) - _ALLOWED_KEYS)
    if unknown_keys:
        raise ValueError(f"{context} contains unsupported keys: {', '.join(unknown_keys)}")


def _parse_mem_agg_params(params: Mapping[str, Any]) -> MemAggParams:
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
    kcenter_mode = str(params.get("kcenter_mode", "budgeted")).strip().lower()
    if kcenter_mode not in {"budgeted", "merge_reduce"}:
        raise ValueError("kcenter_mode must be one of: budgeted, merge_reduce.")
    kcenter_chunk_coreset_size = _require_positive_int(
        value=params.get("kcenter_chunk_coreset_size", 256),
        field_name="kcenter_chunk_coreset_size",
    )
    raw_kcenter_distance_threshold = params.get("kcenter_distance_threshold")
    if raw_kcenter_distance_threshold is None:
        kcenter_distance_threshold: float | None = None
    else:
        if isinstance(raw_kcenter_distance_threshold, bool) or not isinstance(
            raw_kcenter_distance_threshold, (float, int)
        ):
            raise ValueError("kcenter_distance_threshold must be null or float >= 0.")
        kcenter_distance_threshold = float(raw_kcenter_distance_threshold)
        if kcenter_distance_threshold < 0.0:
            raise ValueError("kcenter_distance_threshold must be null or float >= 0.")
    return MemAggParams(
        n_clusters=n_clusters,
        cluster_range=cluster_range,
        streaming_enforce_cluster_budget=streaming_enforce_cluster_budget,
        kcenter_mode=kcenter_mode,
        kcenter_chunk_coreset_size=kcenter_chunk_coreset_size,
        kcenter_distance_threshold=kcenter_distance_threshold,
    )


class MemoryAggParamBindingMixin:
    """Memory-aggregation param binding localized to each concrete plugin."""

    _bound_params: dict[str, Any]
    _bound_bind_context: MemAggBindContextLike
    _mem_agg_params: MemAggParams

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
        if normalized_contract not in {"OFFLINE", "STREAMING"}:
            raise ValueError(
                "bind_context.training_contract must be one of "
                "{'OFFLINE', 'STREAMING'}: "
                f"value={training_contract!r}"
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
