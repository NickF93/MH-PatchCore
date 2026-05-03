"""Plugin-local scoring runtime helpers for `reference_max`."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Protocol, cast

import numpy as np
import scipy.ndimage as ndimage  # type: ignore[import-untyped]
import torch
import torch.nn.functional as F
from ..contracts import MemoryBankPayload, PatchMakerLike, StructuredGlobalDensityBank


class _NNMethodProtocol(Protocol):
    def run(
        self,
        n_nearest_neighbours: int,
        query_features: np.ndarray,
        index_features: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        ...


class _AnomalyScorerProtocol(Protocol):
    detection_features: Any
    nn_method: _NNMethodProtocol | None


def _require_nn_method(
    anomaly_scorer: _AnomalyScorerProtocol,
    *,
    context: str,
) -> _NNMethodProtocol:
    nn_method = getattr(anomaly_scorer, "nn_method", None)
    if nn_method is None:
        raise RuntimeError(f"{context} requires an NN-backed anomaly scorer.")
    return nn_method


class _PNIFaithfulGateStateLike(Protocol):
    grid_height: int
    grid_width: int
    position_prior: np.ndarray
    neighborhood_kernel_size: int
    neighborhood_use_relative: bool
    prior_mix_gamma: float
    faithful_prior_threshold: float
    faithful_distance_scale: float
    assignment_chunk_size: int
    prototype_count: int
    prototype_chunk_size: int


class RescaleSegmentor:
    """Convert patch-score grids to full-resolution segmentation maps."""

    def __init__(
        self,
        *,
        device: torch.device | str,
        target_size: int | tuple[int, int] = 224,
    ) -> None:
        self.device = device
        self.target_size = target_size
        self.smoothing = 4

    def convert_to_segmentation(
        self,
        patch_scores: np.ndarray | torch.Tensor,
    ) -> list[np.ndarray]:
        with torch.no_grad():
            if isinstance(patch_scores, np.ndarray):
                patch_scores = torch.from_numpy(patch_scores)
            _scores = patch_scores.to(self.device)
            _scores = _scores.unsqueeze(1)
            _scores = F.interpolate(
                _scores, size=self.target_size, mode="bilinear", align_corners=False
            )
            _scores = _scores.squeeze(1)
            patch_scores = _scores.cpu().numpy()
        return [
            ndimage.gaussian_filter(patch_score, sigma=self.smoothing)
            for patch_score in patch_scores
        ]


class ScoringRuntimeMixin:
    """Plugin-local runtime factory mixin for scoring plugins."""

    supports_train: bool = True
    supports_inference: bool = True

    def create_segmentor(
        self,
        *,
        device: torch.device | str,
        target_size: int | tuple[int, int],
    ) -> RescaleSegmentor:
        return RescaleSegmentor(
            device=device,
            target_size=target_size,
        )

    def aux_state_fit_start(
        self,
        *,
        memory_bank: MemoryBankPayload,
    ) -> object | None:
        del memory_bank
        return None

    def aux_state_fit_update(
        self,
        *,
        fit_state: object | None,
        features: np.ndarray,
        batch_size: int,
        patch_shape: tuple[int, int],
        locality_context: object | None = None,
    ) -> object | None:
        del features, batch_size, patch_shape, locality_context
        return fit_state

    def aux_state_fit_finalize(
        self,
        *,
        fit_state: object | None,
    ) -> object | None:
        return fit_state

    def aux_state_validate_loaded(
        self,
        *,
        state: object | None,
    ) -> None:
        del state
        return None


def compute_reference_image_scores(
    *,
    patch_maker: PatchMakerLike,
    patch_scores: np.ndarray,
    batchsize: int,
) -> np.ndarray:
    """Compute image scores via max-reduced patch anomaly distances."""
    patch_scores_tensor = torch.as_tensor(patch_scores, dtype=torch.float32)
    image_score_grid = patch_maker.unpatch_scores(
        patch_scores_tensor,
        batchsize=batchsize,
    )
    image_scores_tensor = torch.as_tensor(image_score_grid, dtype=torch.float32)
    image_scores_tensor = image_scores_tensor.reshape(
        *image_scores_tensor.shape[:2],
        -1,
    )
    image_score_values = patch_maker.score(image_scores_tensor)
    return np.asarray(image_score_values, dtype=np.float64)


def compute_reference_topk_soft_image_scores(
    *,
    patch_maker: PatchMakerLike,
    effective_patch_scores: np.ndarray,
    batchsize: int,
    topk_k: int,
    topk_temperature: float,
) -> np.ndarray:
    """Compute image scores from a soft top-k aggregation of patch scores."""
    if int(topk_k) < 1:
        raise ValueError("topk_k must be >= 1 for GLOBAL_TOPK_SOFT image scoring.")
    if float(topk_temperature) <= 0.0:
        raise ValueError(
            "topk_temperature must be > 0 for GLOBAL_TOPK_SOFT image scoring."
        )

    effective_patch_scores_np = np.asarray(effective_patch_scores, dtype=np.float32)
    if effective_patch_scores_np.ndim != 1:
        raise ValueError(
            "effective_patch_scores must be 1D for GLOBAL_TOPK_SOFT image scoring."
        )
    if not np.all(np.isfinite(effective_patch_scores_np)):
        raise ValueError(
            "effective_patch_scores must be finite for GLOBAL_TOPK_SOFT image scoring."
        )

    per_image_effective_scores = _flatten_per_image_patch_scores(
        patch_maker=patch_maker,
        patch_scores=effective_patch_scores_np,
        batchsize=batchsize,
    )
    position_count = int(per_image_effective_scores.shape[1])
    k_eff = min(int(topk_k), position_count)
    if k_eff == 1:
        return compute_reference_image_scores(
            patch_maker=patch_maker,
            patch_scores=effective_patch_scores_np,
            batchsize=batchsize,
        )

    topk_indices = np.argsort(-per_image_effective_scores, axis=1)[:, :k_eff]
    topk_scores = np.take_along_axis(per_image_effective_scores, topk_indices, axis=1)
    scaled_topk_scores = topk_scores / float(topk_temperature)
    scaled_topk_scores = scaled_topk_scores - np.max(
        scaled_topk_scores,
        axis=1,
        keepdims=True,
    )
    weights = np.exp(scaled_topk_scores)
    weight_sums = np.sum(weights, axis=1, keepdims=True)
    if not np.all(np.isfinite(weight_sums)) or np.any(weight_sums <= 0.0):
        raise RuntimeError("GLOBAL_TOPK_SOFT produced invalid top-k softmax weights.")
    normalized_weights = weights / weight_sums
    return np.asarray(
        np.sum(normalized_weights * topk_scores, axis=1, dtype=np.float64),
        dtype=np.float64,
    )


def compute_reference_topp_mean_image_scores(
    *,
    patch_maker: PatchMakerLike,
    effective_patch_scores: np.ndarray,
    batchsize: int,
    topp_p: float,
    topp_max_k: int,
) -> np.ndarray:
    """Compute image scores from a top-p mean aggregation of patch scores."""
    if not 0.0 < float(topp_p) <= 1.0:
        raise ValueError("topp_p must be in (0, 1] for GLOBAL_TOPP_MEAN image scoring.")
    if int(topp_max_k) < 1:
        raise ValueError("topp_max_k must be >= 1 for GLOBAL_TOPP_MEAN image scoring.")

    effective_patch_scores_np = np.asarray(effective_patch_scores, dtype=np.float32)
    if effective_patch_scores_np.ndim != 1:
        raise ValueError(
            "effective_patch_scores must be 1D for GLOBAL_TOPP_MEAN image scoring."
        )
    if not np.all(np.isfinite(effective_patch_scores_np)):
        raise ValueError(
            "effective_patch_scores must be finite for GLOBAL_TOPP_MEAN image scoring."
        )

    per_image_effective_scores = _flatten_per_image_patch_scores(
        patch_maker=patch_maker,
        patch_scores=effective_patch_scores_np,
        batchsize=batchsize,
    )
    position_count = int(per_image_effective_scores.shape[1])
    max_candidates = min(int(topp_max_k), position_count)
    per_image_sorted_indices = np.argsort(-per_image_effective_scores, axis=1)

    image_scores = np.empty(batchsize, dtype=np.float64)
    topp_threshold = float(topp_p)

    for image_idx in range(batchsize):
        sorted_indices = per_image_sorted_indices[image_idx]
        sorted_scores = per_image_effective_scores[image_idx, sorted_indices]
        mass_scores = np.clip(sorted_scores, 0.0, None)
        total_mass = float(np.sum(mass_scores, dtype=np.float64))
        if not np.isfinite(total_mass) or total_mass <= 0.0:
            k_eff = 1
        else:
            cumulative_mass = np.cumsum(mass_scores / total_mass, dtype=np.float64)
            k_mass = int(np.searchsorted(cumulative_mass, topp_threshold, side="left")) + 1
            k_eff = min(max(1, k_mass), max_candidates)
        image_scores[image_idx] = float(np.mean(sorted_scores[:k_eff], dtype=np.float64))

    return image_scores


def compute_paper_eq7_image_scores(
    *,
    patch_maker: PatchMakerLike,
    anomaly_scorer: _AnomalyScorerProtocol,
    paper_reweight_num_nn: int,
    features: np.ndarray,
    query_distances: np.ndarray,
    query_nns: np.ndarray,
    batchsize: int,
) -> np.ndarray:
    """Compute image scores using PatchCore Eq.7 neighborhood reweighting."""
    if query_distances.ndim != 2:
        raise ValueError("query_distances must be 2D for PAPER_EQ7 image scoring.")
    if query_nns.ndim != 2:
        raise ValueError("query_nns must be 2D for PAPER_EQ7 image scoring.")
    if query_distances.shape != query_nns.shape:
        raise ValueError(
            "query_distances and query_nns must have identical shapes for PAPER_EQ7 image scoring."
        )
    if query_distances.shape[1] < 1:
        raise ValueError(
            "query_distances must include at least one nearest neighbor for PAPER_EQ7 image scoring."
        )
    if features.shape[0] != query_distances.shape[0]:
        raise ValueError(
            "features and query_distances must have matching first dimension for PAPER_EQ7 image scoring."
        )

    memory_bank = getattr(anomaly_scorer, "detection_features", None)
    if not isinstance(memory_bank, np.ndarray):
        raise RuntimeError(
            "PAPER_EQ7 image scoring requires in-memory detection_features. "
            "Re-fit the model or load a checkpoint saved with detection features."
        )
    if memory_bank.ndim != 2:
        raise RuntimeError(
            "PAPER_EQ7 image scoring requires detection_features with shape [N, D]."
        )
    if memory_bank.shape[0] == 0:
        raise RuntimeError(
            "PAPER_EQ7 image scoring requires a non-empty detection_features memory bank."
        )

    nearest_patch_scores = np.asarray(query_distances[:, 0], dtype=np.float32)
    nearest_patch_scores_tensor = torch.as_tensor(
        nearest_patch_scores,
        dtype=torch.float32,
    )
    per_image_patch_score_grid = patch_maker.unpatch_scores(
        nearest_patch_scores_tensor,
        batchsize=batchsize,
    )
    per_image_patch_scores = np.asarray(
        per_image_patch_score_grid,
        dtype=np.float64,
    ).reshape(batchsize, -1)
    per_image_max_indices = np.argmax(per_image_patch_scores, axis=1)
    per_image_max_scores = per_image_patch_scores[
        np.arange(batchsize),
        per_image_max_indices,
    ]

    num_patches_per_image = int(per_image_patch_scores.shape[1])
    global_patch_indices = (
        np.arange(batchsize, dtype=np.int64) * num_patches_per_image
        + per_image_max_indices.astype(np.int64)
    )
    nearest_memory_indices = query_nns[global_patch_indices, 0].astype(np.int64)

    reweight_k = min(paper_reweight_num_nn, int(memory_bank.shape[0]))
    if reweight_k <= 1:
        return per_image_max_scores

    query_patch_features = np.asarray(features[global_patch_indices], dtype=np.float32)
    anchor_features = np.asarray(memory_bank[nearest_memory_indices], dtype=np.float32)
    _, support_neighbor_indices = _require_nn_method(
        anomaly_scorer,
        context="REFERENCE_MAX Eq.7-style image scoring",
    ).run(
        n_nearest_neighbours=reweight_k,
        query_features=anchor_features,
        index_features=np.asarray(memory_bank, dtype=np.float32),
    )

    weighted_scores: np.ndarray = np.empty(batchsize, dtype=np.float64)
    for image_idx in range(batchsize):
        support_indices = support_neighbor_indices[image_idx].astype(np.int64)
        support_features = np.asarray(memory_bank[support_indices], dtype=np.float64)
        query_feature = np.asarray(query_patch_features[image_idx], dtype=np.float64)

        support_sq_distances = np.sum((support_features - query_feature) ** 2, axis=1)
        if support_sq_distances.size <= 1:
            weighted_scores[image_idx] = per_image_max_scores[image_idx]
            continue

        max_distance = float(np.max(support_sq_distances))
        logits = np.exp(support_sq_distances - max_distance)
        denominator = float(np.sum(logits))
        numerator = float(np.exp(per_image_max_scores[image_idx] - max_distance))
        if denominator <= 0.0 or not np.isfinite(denominator):
            weighted_scores[image_idx] = per_image_max_scores[image_idx]
            continue

        weight = 1.0 - (numerator / denominator)
        weight = float(np.clip(weight, 0.0, 1.0))
        weighted_scores[image_idx] = weight * per_image_max_scores[image_idx]

    return weighted_scores


@dataclass(frozen=True)
class PatchScoringSelection:
    """Selection inputs for patch-scoring strategy factory."""

    mode: str
    pni_state: object | None


class PatchScoringStrategy(ABC):
    """Contract for patch-scoring strategies."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable strategy name."""

    @abstractmethod
    def score(
        self,
        *,
        features: np.ndarray,
        patch_scores: np.ndarray,
        query_nns: np.ndarray,
        memory_bank: np.ndarray | None,
        batch_size: int,
        patch_shape: tuple[int, int],
    ) -> np.ndarray:
        """Compute per-patch anomaly scores."""


class GlobalOnlyPatchScoringStrategy(PatchScoringStrategy):
    """Default PatchCore behavior: keep nearest-neighbor patch scores."""

    @property
    def name(self) -> str:
        return "GLOBAL_ONLY"

    def score(
        self,
        *,
        features: np.ndarray,
        patch_scores: np.ndarray,
        query_nns: np.ndarray,
        memory_bank: np.ndarray | None,
        batch_size: int,
        patch_shape: tuple[int, int],
    ) -> np.ndarray:
        _ = features, query_nns, memory_bank, batch_size, patch_shape
        scores = np.asarray(patch_scores, dtype=np.float32)
        if scores.ndim != 1:
            raise ValueError(
                "GlobalOnlyPatchScoringStrategy expects 1D patch_scores, "
                f"got shape={scores.shape}."
            )
        if not np.all(np.isfinite(scores)):
            raise ValueError("GlobalOnlyPatchScoringStrategy received non-finite scores.")
        return scores


class GlobalTopKSoftPatchScoringStrategy(GlobalOnlyPatchScoringStrategy):
    """Top-k soft mode keeps the default global per-patch score semantics."""

    @property
    def name(self) -> str:
        return "GLOBAL_TOPK_SOFT"


class GlobalTopPMeanPatchScoringStrategy(GlobalOnlyPatchScoringStrategy):
    """Top-p mean mode keeps the default global per-patch score semantics."""

    @property
    def name(self) -> str:
        return "GLOBAL_TOPP_MEAN"


def _validate_2d_matrix(matrix: np.ndarray, *, name: str) -> np.ndarray:
    """Validate and normalize a 2D finite matrix."""
    matrix_np = np.asarray(matrix)
    if matrix_np.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape={matrix_np.shape}.")
    if matrix_np.shape[0] == 0 or matrix_np.shape[1] == 0:
        raise ValueError(f"{name} must be non-empty, got shape={matrix_np.shape}.")
    if not np.all(np.isfinite(matrix_np)):
        raise ValueError(f"{name} contains non-finite values.")
    return matrix_np.astype(np.float32, copy=False)


def _validate_chunked_distance_inputs(
    *,
    queries: np.ndarray,
    prototypes: np.ndarray,
    query_chunk_size: int,
    prototype_chunk_size: int | None,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Validate chunked distance inputs and resolve effective chunk sizes."""
    queries_np = _validate_2d_matrix(queries, name="queries")
    prototypes_np = _validate_2d_matrix(prototypes, name="prototypes")
    if queries_np.shape[1] != prototypes_np.shape[1]:
        raise ValueError(
            "queries and prototypes must have the same feature dimension; "
            f"got {queries_np.shape[1]} and {prototypes_np.shape[1]}."
        )
    if (
        isinstance(query_chunk_size, bool)
        or not isinstance(query_chunk_size, int)
        or query_chunk_size <= 0
    ):
        raise ValueError("query_chunk_size must be a positive integer.")
    if prototype_chunk_size is None:
        prototype_chunk = int(prototypes_np.shape[0])
    elif (
        isinstance(prototype_chunk_size, bool)
        or not isinstance(prototype_chunk_size, int)
        or prototype_chunk_size <= 0
    ):
        raise ValueError("prototype_chunk_size must be a positive integer when provided.")
    else:
        prototype_chunk = int(prototype_chunk_size)
    return queries_np, prototypes_np, int(query_chunk_size), prototype_chunk


def iterate_chunked_l2_distance_blocks(
    queries: np.ndarray,
    prototypes: np.ndarray,
    *,
    query_chunk_size: int,
    prototype_chunk_size: int | None = None,
) -> Iterator[tuple[slice, slice, np.ndarray]]:
    """Yield exact squared L2 distance blocks over query/prototype chunks."""
    queries_np, prototypes_np, query_chunk, proto_chunk = _validate_chunked_distance_inputs(
        queries=queries,
        prototypes=prototypes,
        query_chunk_size=query_chunk_size,
        prototype_chunk_size=prototype_chunk_size,
    )
    proto_norms_full = np.asarray(
        np.sum(prototypes_np * prototypes_np, axis=1, dtype=np.float32),
        dtype=np.float32,
    )
    n_queries = int(queries_np.shape[0])
    n_prototypes = int(prototypes_np.shape[0])
    for q_start in range(0, n_queries, query_chunk):
        q_end = min(q_start + query_chunk, n_queries)
        query_block = queries_np[q_start:q_end]
        query_norms = np.asarray(
            np.sum(query_block * query_block, axis=1, dtype=np.float32, keepdims=True),
            dtype=np.float32,
        )
        for p_start in range(0, n_prototypes, proto_chunk):
            p_end = min(p_start + proto_chunk, n_prototypes)
            prototype_block = prototypes_np[p_start:p_end]
            dots = query_block @ prototype_block.T
            distance_block = (
                query_norms
                + proto_norms_full[p_start:p_end].reshape(1, -1)
                - (2.0 * dots)
            )
            np.maximum(distance_block, 0.0, out=distance_block)
            yield (
                slice(q_start, q_end),
                slice(p_start, p_end),
                distance_block.astype(np.float32, copy=False),
            )


def _validate_patch_shape(
    *,
    patch_shape: tuple[int, int],
    batch_size: int,
    n_patches: int,
) -> tuple[int, int, int]:
    """Validate patch-grid metadata against flattened patch count."""
    if len(patch_shape) != 2:
        raise ValueError(f"patch_shape must contain two integers, got {patch_shape}.")
    h, w = int(patch_shape[0]), int(patch_shape[1])
    if h <= 0 or w <= 0:
        raise ValueError(f"patch_shape values must be > 0, got {patch_shape}.")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")
    per_image = int(h * w)
    expected = int(batch_size * per_image)
    if n_patches != expected:
        raise ValueError(
            "Patch scoring received inconsistent patch count: "
            f"expected {expected} from batch_size={batch_size} and patch_shape={patch_shape}, "
            f"got {n_patches}."
        )
    return h, w, per_image


def _build_neighborhood_indices(
    *,
    height: int,
    width: int,
    kernel_size: int,
    use_relative: bool,
) -> list[np.ndarray]:
    """Build per-position neighbor index lists for a fixed patch grid."""
    radius = kernel_size // 2
    neighbors: list[np.ndarray] = []
    for r in range(height):
        for c in range(width):
            current: list[int] = []
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    if use_relative and dr == 0 and dc == 0:
                        continue
                    rr = r + dr
                    cc = c + dc
                    if rr < 0 or rr >= height or cc < 0 or cc >= width:
                        continue
                    current.append(int(rr * width + cc))
            if not current:
                current = [int(r * width + c)]
            neighbors.append(np.asarray(current, dtype=np.int64))
    return neighbors


def _compute_neighborhood_prior(
    *,
    labels_flat: np.ndarray,
    n_classes: int,
    neighbor_indices: list[np.ndarray],
) -> np.ndarray:
    """Estimate local class priors from nearest-prototype labels."""
    priors = np.zeros((len(neighbor_indices), n_classes), dtype=np.float32)
    for position, indices in enumerate(neighbor_indices):
        neighborhood_labels = labels_flat[indices]
        counts = np.bincount(neighborhood_labels, minlength=n_classes).astype(
            np.float32,
            copy=False,
        )
        denom = float(np.sum(counts, dtype=np.float32))
        if denom <= 0.0:
            raise RuntimeError("Neighborhood prior encountered zero denominator.")
        priors[position] = counts / denom
    return priors


def _flatten_per_image_patch_scores(
    *,
    patch_maker: PatchMakerLike,
    patch_scores: np.ndarray | torch.Tensor,
    batchsize: int,
) -> np.ndarray:
    """Reshape flat patch scores to one `[batch, positions]` matrix."""
    patch_scores_tensor = torch.as_tensor(patch_scores, dtype=torch.float32)
    per_image_patch_score_grid = patch_maker.unpatch_scores(
        patch_scores_tensor,
        batchsize=batchsize,
    )
    return np.asarray(
        per_image_patch_score_grid,
        dtype=np.float64,
    ).reshape(batchsize, -1)


class PNIFaithfulGatePatchScoringStrategy(PatchScoringStrategy):
    """PNI faithful-gate scoring with position and neighborhood priors."""

    def __init__(self, state: object) -> None:
        required_fields = (
            "grid_height",
            "grid_width",
            "position_prior",
            "neighborhood_kernel_size",
            "neighborhood_use_relative",
            "prior_mix_gamma",
            "faithful_prior_threshold",
            "faithful_distance_scale",
            "assignment_chunk_size",
            "prototype_count",
        )
        if any(not hasattr(state, field_name) for field_name in required_fields):
            raise ValueError("PNI faithful strategy requires PNIFaithfulGateState.")
        self._state = cast(_PNIFaithfulGateStateLike, state)

        position_prior = np.asarray(getattr(state, "position_prior"))
        if position_prior.ndim != 3:
            raise ValueError(
                "PNI faithful state position_prior must be 3D [H,W,K], "
                f"got shape={position_prior.shape}."
            )
        if (
            int(position_prior.shape[0]) != int(getattr(state, "grid_height"))
            or int(position_prior.shape[1]) != int(getattr(state, "grid_width"))
            or int(position_prior.shape[2]) != int(getattr(state, "prototype_count"))
        ):
            raise ValueError(
                "PNI faithful state is inconsistent: grid/prototype metadata "
                "does not match position_prior shape."
            )
        self._position_prior_flat = position_prior.reshape(
            int(getattr(state, "grid_height") * getattr(state, "grid_width")),
            int(getattr(state, "prototype_count")),
        ).astype(np.float32, copy=False)
        self._neighbor_indices = _build_neighborhood_indices(
            height=int(getattr(state, "grid_height")),
            width=int(getattr(state, "grid_width")),
            kernel_size=int(getattr(state, "neighborhood_kernel_size")),
            use_relative=bool(getattr(state, "neighborhood_use_relative")),
        )
        raw_prototype_chunk = int(getattr(state, "prototype_chunk_size", 0))
        self._prototype_chunk_size = (
            int(getattr(state, "prototype_count"))
            if raw_prototype_chunk <= 0
            else raw_prototype_chunk
        )

    @property
    def name(self) -> str:
        return "PNI_FAITHFUL_GATE"

    def score(
        self,
        *,
        features: np.ndarray,
        patch_scores: np.ndarray,
        query_nns: np.ndarray,
        memory_bank: np.ndarray | None,
        batch_size: int,
        patch_shape: tuple[int, int],
    ) -> np.ndarray:
        """Compute faithful-gate PNI scores for one flattened patch batch."""
        del patch_scores
        features_np = np.asarray(features, dtype=np.float32)
        if features_np.ndim != 2:
            raise ValueError(
                "PNI faithful strategy expects 2D features [N,D], "
                f"got shape={features_np.shape}."
            )
        if not np.all(np.isfinite(features_np)):
            raise ValueError("PNI faithful strategy received non-finite features.")
        if memory_bank is None:
            raise RuntimeError("PNI faithful strategy requires a fitted memory bank.")
        memory_bank_np = np.asarray(memory_bank, dtype=np.float32)
        if memory_bank_np.ndim != 2:
            raise ValueError(
                "PNI faithful strategy expects 2D memory bank [K,D], "
                f"got shape={memory_bank_np.shape}."
            )
        if int(memory_bank_np.shape[0]) != int(getattr(self._state, "prototype_count")):
            raise ValueError(
                "PNI faithful strategy received memory bank with unexpected size: "
                f"expected {getattr(self._state, 'prototype_count')}, got {memory_bank_np.shape[0]}."
            )
        if int(features_np.shape[1]) != int(memory_bank_np.shape[1]):
            raise ValueError(
                "PNI faithful strategy feature dimension mismatch: "
                f"features={features_np.shape[1]}, memory_bank={memory_bank_np.shape[1]}."
            )
        if not np.all(np.isfinite(memory_bank_np)):
            raise ValueError("PNI faithful strategy received non-finite memory bank.")

        query_nns_np = np.asarray(query_nns)
        if query_nns_np.ndim != 2 or int(query_nns_np.shape[1]) < 1:
            raise ValueError(
                "PNI faithful strategy expects query_nns with shape [N, >=1], "
                f"got shape={query_nns_np.shape}."
            )
        nearest_labels = query_nns_np[:, 0].astype(np.int64, copy=False)
        if np.any(nearest_labels < 0) or np.any(nearest_labels >= memory_bank_np.shape[0]):
            raise ValueError("PNI faithful strategy received invalid nearest-neighbor labels.")

        h, w, per_image = _validate_patch_shape(
            patch_shape=patch_shape,
            batch_size=batch_size,
            n_patches=int(features_np.shape[0]),
        )
        if h != int(getattr(self._state, "grid_height")) or w != int(
            getattr(self._state, "grid_width")
        ):
            raise ValueError(
                "PNI faithful strategy patch grid mismatch: "
                f"state={(getattr(self._state, 'grid_height'), getattr(self._state, 'grid_width'))} "
                f"predict={(h, w)}."
            )

        output_scores = np.empty(int(features_np.shape[0]), dtype=np.float32)
        gamma = float(getattr(self._state, "prior_mix_gamma"))
        threshold = float(getattr(self._state, "faithful_prior_threshold"))
        scale = float(getattr(self._state, "faithful_distance_scale"))
        prototype_count = int(getattr(self._state, "prototype_count"))

        for image_idx in range(batch_size):
            start = int(image_idx * per_image)
            end = int(start + per_image)
            image_labels = nearest_labels[start:end]

            neighborhood_prior = _compute_neighborhood_prior(
                labels_flat=image_labels,
                n_classes=prototype_count,
                neighbor_indices=self._neighbor_indices,
            )
            mixed_prior = (gamma * self._position_prior_flat) + (
                (1.0 - gamma) * neighborhood_prior
            )
            eligible = mixed_prior >= threshold

            # Exact two-dimensional chunking avoids materializing the full
            # [N_patches x N_prototypes] matrix while preserving the same minima.
            image_features = features_np[start:end]
            global_min = np.full(per_image, np.float32(np.inf), dtype=np.float32)
            eligible_min = np.full(per_image, np.float32(np.inf), dtype=np.float32)
            for q_slice, p_slice, distance_block in iterate_chunked_l2_distance_blocks(
                image_features,
                memory_bank_np,
                query_chunk_size=int(getattr(self._state, "assignment_chunk_size")),
                prototype_chunk_size=self._prototype_chunk_size,
            ):
                block_min = np.min(distance_block, axis=1)
                current_global = global_min[q_slice]
                np.minimum(current_global, block_min, out=current_global)
                global_min[q_slice] = current_global

                eligible_block = eligible[q_slice, p_slice]
                if not np.any(eligible_block):
                    continue
                masked_block = np.where(eligible_block, distance_block, np.float32(np.inf))
                block_eligible_min = np.min(masked_block, axis=1)
                current_eligible = eligible_min[q_slice]
                np.minimum(current_eligible, block_eligible_min, out=current_eligible)
                eligible_min[q_slice] = current_eligible

            selected = np.where(np.isfinite(eligible_min), eligible_min, global_min)
            output_scores[start:end] = (scale * selected).astype(np.float32, copy=False)

        if not np.all(np.isfinite(output_scores)):
            raise RuntimeError("PNI faithful strategy produced non-finite patch scores.")
        return output_scores


def create_patch_scoring_strategy(
    selection: PatchScoringSelection,
) -> PatchScoringStrategy:
    """Create patch-scoring strategy from validated selection inputs."""
    mode = str(selection.mode).strip().upper()
    if mode == "GLOBAL_ONLY":
        return GlobalOnlyPatchScoringStrategy()
    if mode == "GLOBAL_TOPK_SOFT":
        return GlobalTopKSoftPatchScoringStrategy()
    if mode == "GLOBAL_TOPP_MEAN":
        return GlobalTopPMeanPatchScoringStrategy()
    if mode == "PNI_FAITHFUL_GATE":
        if selection.pni_state is None:
            raise RuntimeError(
                "PNI_FAITHFUL_GATE requires a fitted PNI state, but none was provided."
            )
        return PNIFaithfulGatePatchScoringStrategy(selection.pni_state)
    if mode == "PNI_SOFT_FUSION":
        raise NotImplementedError(
            "PNI_SOFT_FUSION is defined in configuration as an extension point "
            "but is not implemented in this increment."
        )
    raise ValueError(f"Unsupported patch scoring mode: {selection.mode!r}.")


def _compute_density_effective_patch_scores(
    *,
    patch_scores: np.ndarray,
    patch_scoring_mode: str,
) -> np.ndarray:
    """Apply the supported density-bank reference-max modes without NN assumptions."""
    normalized_mode = str(patch_scoring_mode).strip().upper()
    if normalized_mode not in {"GLOBAL_ONLY", "GLOBAL_TOPK_SOFT", "GLOBAL_TOPP_MEAN"}:
        raise RuntimeError(
            "Structured global density banks are supported by reference_max only for "
            "patch_scoring.mode in {'GLOBAL_ONLY', 'GLOBAL_TOPK_SOFT', 'GLOBAL_TOPP_MEAN'}; "
            f"got {patch_scoring_mode!r}."
        )
    scores = np.asarray(patch_scores, dtype=np.float32)
    if scores.ndim != 1:
        raise ValueError(
            "Density-backed reference_max expects 1D patch_scores, "
            f"got shape={scores.shape}."
        )
    if not np.all(np.isfinite(scores)):
        raise ValueError("Density-backed reference_max received non-finite patch_scores.")
    return scores


def _compute_effective_patch_scores(
    *,
    features: np.ndarray,
    patch_scores: np.ndarray,
    query_nns: np.ndarray,
    patch_shape: tuple[int, int],
    batchsize: int,
    anomaly_scorer: object,
    patch_scoring_mode: str,
    patch_scoring_state: object | None,
) -> np.ndarray:
    detection_features = getattr(anomaly_scorer, "detection_features", None)
    if isinstance(detection_features, StructuredGlobalDensityBank):
        return _compute_density_effective_patch_scores(
            patch_scores=patch_scores,
            patch_scoring_mode=patch_scoring_mode,
        )

    scoring_strategy = create_patch_scoring_strategy(
        PatchScoringSelection(
            mode=patch_scoring_mode,
            pni_state=patch_scoring_state,
        )
    )
    memory_bank = np.asarray(detection_features)
    return scoring_strategy.score(
        features=features,
        patch_scores=np.asarray(patch_scores),
        query_nns=np.asarray(query_nns),
        memory_bank=(
            memory_bank if memory_bank.ndim == 2 and memory_bank.shape[0] > 0 else None
        ),
        batch_size=batchsize,
        patch_shape=patch_shape,
    )


def score_reference_max(
    *,
    features: np.ndarray,
    patch_scores: np.ndarray,
    query_nns: np.ndarray,
    patch_shape: tuple[int, int],
    batchsize: int,
    patch_maker: PatchMakerLike,
    anomaly_scorer: object,
    patch_scoring_mode: str,
    patch_scoring_state: object | None,
    topk_k: int,
    topk_temperature: float,
    topp_p: float,
    topp_max_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    effective_patch_scores = _compute_effective_patch_scores(
        features=features,
        patch_scores=patch_scores,
        query_nns=query_nns,
        patch_shape=patch_shape,
        batchsize=batchsize,
        anomaly_scorer=anomaly_scorer,
        patch_scoring_mode=patch_scoring_mode,
        patch_scoring_state=patch_scoring_state,
    )
    normalized_mode = str(patch_scoring_mode).strip().upper()
    if normalized_mode == "GLOBAL_TOPK_SOFT":
        image_scores = compute_reference_topk_soft_image_scores(
            patch_maker=patch_maker,
            effective_patch_scores=effective_patch_scores,
            batchsize=batchsize,
            topk_k=topk_k,
            topk_temperature=topk_temperature,
        )
    elif normalized_mode == "GLOBAL_TOPP_MEAN":
        image_scores = compute_reference_topp_mean_image_scores(
            patch_maker=patch_maker,
            effective_patch_scores=effective_patch_scores,
            batchsize=batchsize,
            topp_p=topp_p,
            topp_max_k=topp_max_k,
        )
    else:
        image_scores = compute_reference_image_scores(
            patch_maker=patch_maker,
            patch_scores=effective_patch_scores,
            batchsize=batchsize,
        )
    return image_scores, effective_patch_scores


def score_paper_eq7(
    *,
    features: np.ndarray,
    patch_scores: np.ndarray,
    query_distances: np.ndarray,
    query_nns: np.ndarray,
    patch_shape: tuple[int, int],
    batchsize: int,
    patch_maker: PatchMakerLike,
    anomaly_scorer: object,
    patch_scoring_mode: str,
    patch_scoring_state: object | None,
    paper_reweight_num_nn: int,
) -> tuple[np.ndarray, np.ndarray]:
    effective_patch_scores = _compute_effective_patch_scores(
        features=features,
        patch_scores=patch_scores,
        query_nns=query_nns,
        patch_shape=patch_shape,
        batchsize=batchsize,
        anomaly_scorer=anomaly_scorer,
        patch_scoring_mode=patch_scoring_mode,
        patch_scoring_state=patch_scoring_state,
    )
    image_scores = compute_paper_eq7_image_scores(
        patch_maker=patch_maker,
        anomaly_scorer=cast(_AnomalyScorerProtocol, anomaly_scorer),
        paper_reweight_num_nn=paper_reweight_num_nn,
        features=features,
        query_distances=query_distances,
        query_nns=query_nns,
        batchsize=batchsize,
    )
    return image_scores, effective_patch_scores


__all__ = [
    "RescaleSegmentor",
    "ScoringRuntimeMixin",
    "compute_reference_image_scores",
    "compute_reference_topk_soft_image_scores",
    "compute_reference_topp_mean_image_scores",
    "compute_paper_eq7_image_scores",
    "score_reference_max",
    "score_paper_eq7",
]
