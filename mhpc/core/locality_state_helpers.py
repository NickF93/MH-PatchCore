"""Generic helper surface for structured bank/query payload contracts."""

from __future__ import annotations

import typing as _T

import numpy as np

from .plugins.distance.contracts import DistanceAnomalyScorer
from .plugins.locality_state_contract import (
    DistanceQueryPayload,
    MemoryBankPayload,
    StructuredGlobalDensityBank,
    StructuredGlobalDensityQueryResult,
    StructuredGlobalNNBank,
    StructuredLocalMemoryBank,
    StructuredLocalQueryResult,
)


def normalize_structured_global_density_bank(
    memory_bank: StructuredGlobalDensityBank,
) -> StructuredGlobalDensityBank:
    """Validate and normalize structured global density-bank payloads."""
    model_family = str(memory_bank.model_family).strip()
    if not model_family:
        raise TypeError("Structured global density bank requires a non-empty model_family.")
    if isinstance(memory_bank.feature_dim, bool) or int(memory_bank.feature_dim) <= 0:
        raise TypeError("Structured global density bank requires feature_dim > 0.")
    feature_dim = int(memory_bank.feature_dim)
    covariance_type = str(memory_bank.covariance_type).strip()
    if not covariance_type:
        raise TypeError("Structured global density bank requires a non-empty covariance_type.")

    component_weights = np.asarray(memory_bank.component_weights, dtype=np.float64)
    component_means = np.asarray(memory_bank.component_means, dtype=np.float64)
    component_variances = np.asarray(memory_bank.component_variances, dtype=np.float64)
    component_effective_counts = np.asarray(
        memory_bank.component_effective_counts,
        dtype=np.float64,
    )
    if component_weights.ndim != 1:
        raise TypeError(
            "Structured global density bank requires 1D component_weights; "
            f"got {component_weights.shape}."
        )
    component_count = int(component_weights.shape[0])
    if component_count <= 0:
        raise TypeError(
            "Structured global density bank requires at least one component."
        )
    if component_means.ndim != 2:
        raise TypeError(
            "Structured global density bank requires 2D component_means; "
            f"got {component_means.shape}."
        )
    if component_means.shape != (component_count, feature_dim):
        raise TypeError(
            "Structured global density bank requires component_means with shape "
            f"[{component_count}, {feature_dim}]; got {component_means.shape}."
        )
    if component_effective_counts.shape != (component_count,):
        raise TypeError(
            "Structured global density bank requires 1D component_effective_counts "
            f"with length {component_count}; got {component_effective_counts.shape}."
        )
    if component_variances.ndim == 1:
        if component_variances.shape != (component_count,):
            raise TypeError(
                "Structured global density bank requires 1D component_variances "
                f"with length {component_count}; got {component_variances.shape}."
            )
    elif component_variances.ndim == 2:
        if component_variances.shape != (component_count, feature_dim):
            raise TypeError(
                "Structured global density bank requires 2D component_variances "
                f"with shape [{component_count}, {feature_dim}]; got "
                f"{component_variances.shape}."
            )
    else:
        raise TypeError(
            "Structured global density bank requires 1D or 2D component_variances; "
            f"got {component_variances.shape}."
        )

    arrays_to_check = {
        "component_weights": component_weights,
        "component_means": component_means,
        "component_variances": component_variances,
        "component_effective_counts": component_effective_counts,
    }
    for field_name, field_value in arrays_to_check.items():
        if not np.all(np.isfinite(field_value)):
            raise TypeError(
                "Structured global density bank requires finite values for "
                f"{field_name}."
            )
    if np.any(component_weights < 0.0):
        raise TypeError(
            "Structured global density bank requires non-negative component_weights."
        )
    total_weight = float(np.sum(component_weights, dtype=np.float64))
    if total_weight <= 0.0:
        raise TypeError(
            "Structured global density bank requires positive total component weight."
        )
    if np.any(component_effective_counts < 0.0):
        raise TypeError(
            "Structured global density bank requires non-negative "
            "component_effective_counts."
        )
    if np.any(component_variances <= 0.0):
        raise TypeError(
            "Structured global density bank requires strictly positive "
            "component_variances."
        )
    regularization = float(memory_bank.regularization)
    if not np.isfinite(regularization) or regularization < 0.0:
        raise TypeError(
            "Structured global density bank requires regularization >= 0."
        )
    if isinstance(memory_bank.seen_samples, bool) or int(memory_bank.seen_samples) < 0:
        raise TypeError(
            "Structured global density bank requires seen_samples >= 0."
        )
    if isinstance(memory_bank.update_count, bool) or int(memory_bank.update_count) < 0:
        raise TypeError(
            "Structured global density bank requires update_count >= 0."
        )
    if not isinstance(memory_bank.is_initialized, bool):
        raise TypeError(
            "Structured global density bank requires is_initialized to be boolean."
        )

    return StructuredGlobalDensityBank(
        model_family=model_family,
        component_weights=component_weights,
        component_means=component_means,
        component_variances=component_variances,
        component_effective_counts=component_effective_counts,
        feature_dim=feature_dim,
        covariance_type=covariance_type,
        regularization=regularization,
        seen_samples=int(memory_bank.seen_samples),
        update_count=int(memory_bank.update_count),
        is_initialized=bool(memory_bank.is_initialized),
    )


def normalize_structured_global_nn_bank(
    memory_bank: StructuredGlobalNNBank,
) -> StructuredGlobalNNBank:
    """Validate and normalize structured global NN-bank payloads."""
    features = np.ascontiguousarray(np.asarray(memory_bank.features, dtype=np.float32))
    if features.ndim != 2:
        raise TypeError(
            "Structured global NN bank requires 2D features; "
            f"got {features.shape}."
        )
    bank_size = int(features.shape[0])
    feature_dim = int(features.shape[1])
    if bank_size <= 0 or feature_dim <= 0:
        raise TypeError(
            "Structured global NN bank requires features with shape [N, D] "
            f"where N > 0 and D > 0; got {features.shape}."
        )
    if not np.all(np.isfinite(features)):
        raise TypeError("Structured global NN bank requires finite feature values.")

    if memory_bank.self_distances is None:
        return StructuredGlobalNNBank(features=features, self_distances=None)

    self_distances = np.asarray(memory_bank.self_distances, dtype=np.float64)
    if self_distances.ndim != 1 or int(self_distances.shape[0]) != bank_size:
        raise TypeError(
            "Structured global NN bank requires self_distances with shape "
            f"[{bank_size}]; got {self_distances.shape}."
        )
    if not np.all(np.isfinite(self_distances)):
        raise TypeError(
            "Structured global NN bank requires finite self_distances values."
        )
    if np.any(self_distances < 0.0):
        raise TypeError(
            "Structured global NN bank requires non-negative self_distances."
        )
    return StructuredGlobalNNBank(
        features=features,
        self_distances=self_distances,
    )


def normalize_memory_bank_payload(memory_bank: MemoryBankPayload) -> MemoryBankPayload:
    """Normalize materialized banks while preserving structured payloads."""
    if isinstance(memory_bank, np.ndarray):
        return np.asarray(memory_bank, dtype=np.float32)
    if isinstance(memory_bank, StructuredGlobalNNBank):
        return normalize_structured_global_nn_bank(memory_bank)
    if isinstance(memory_bank, StructuredGlobalDensityBank):
        return normalize_structured_global_density_bank(memory_bank)
    if isinstance(memory_bank, StructuredLocalMemoryBank):
        return memory_bank
    raise TypeError(
        "Materialize stage must emit either a global ndarray bank, "
        "StructuredGlobalNNBank, StructuredGlobalDensityBank, or "
        "StructuredLocalMemoryBank."
    )


def count_memory_bank_references(memory_bank: MemoryBankPayload) -> int:
    """Count materialized reference vectors for generic fit/load logging."""
    if isinstance(memory_bank, np.ndarray):
        memory_bank_np = np.asarray(memory_bank)
        if memory_bank_np.ndim != 2:
            raise RuntimeError(
                "Global materialized memory bank must have shape [N, D]; "
                f"got {memory_bank_np.shape}."
            )
        return int(memory_bank_np.shape[0])
    if isinstance(memory_bank, StructuredGlobalNNBank):
        normalized_nn_bank = normalize_structured_global_nn_bank(memory_bank)
        return int(normalized_nn_bank.features.shape[0])
    if isinstance(memory_bank, StructuredGlobalDensityBank):
        normalized_density_bank = normalize_structured_global_density_bank(memory_bank)
        return int(normalized_density_bank.component_weights.shape[0])
    if not isinstance(memory_bank, StructuredLocalMemoryBank):
        raise TypeError(
            "Unsupported detection_features payload type: "
            f"{type(memory_bank).__name__}."
        )
    total = 0
    for bank_entry in memory_bank.position_banks:
        bank_features = np.asarray(bank_entry.features)
        if bank_features.ndim != 2:
            raise RuntimeError(
                "Structured local materialized bank entries must have shape [N, D]; "
                f"got {bank_features.shape} at position={bank_entry.position}."
            )
        total += int(bank_features.shape[0])
    return total


def validate_anomaly_scorer_memory_bank_compatibility(
    *,
    anomaly_scorer: DistanceAnomalyScorer,
    memory_bank: MemoryBankPayload,
    stage: str,
) -> None:
    """Reject bank/scorer pairings that are invalid under the active contracts."""
    if (
        isinstance(memory_bank, StructuredGlobalDensityBank)
        and getattr(anomaly_scorer, "nn_method", None) is not None
    ):
        raise RuntimeError(
            "StructuredGlobalDensityBank cannot be consumed by an NN-backed anomaly "
            f"scorer during {stage}. A density-aware non-NN distance runtime is required."
        )


def resolve_anomaly_scorer_memory_bank(
    *,
    anomaly_scorer: DistanceAnomalyScorer,
    stage: str,
    expected_memory_bank: MemoryBankPayload | None = None,
) -> MemoryBankPayload:
    """Normalize scorer-owned detection_features and validate bank-family stability."""
    detection_features = getattr(anomaly_scorer, "detection_features", None)
    if expected_memory_bank is None:
        memory_bank = require_normalized_detection_features_payload(
            detection_features=detection_features,
            stage=stage,
        )
    else:
        memory_bank = validate_fitted_detection_features_payload(
            expected_memory_bank=expected_memory_bank,
            actual_detection_features=detection_features,
            stage=stage,
        )
    validate_anomaly_scorer_memory_bank_compatibility(
        anomaly_scorer=anomaly_scorer,
        memory_bank=memory_bank,
        stage=stage,
    )
    return memory_bank


def is_supported_detection_features_payload(value: object) -> bool:
    """Return whether a loaded scorer payload matches the active bank contracts."""
    return isinstance(
        value,
        (
            np.ndarray,
            StructuredGlobalNNBank,
            StructuredGlobalDensityBank,
            StructuredLocalMemoryBank,
        ),
    )


def describe_payload_family(value: object) -> str:
    """Return a compact family label for supported bank/query payload validation."""
    if isinstance(value, np.ndarray):
        return "global ndarray memory bank"
    if isinstance(value, StructuredGlobalNNBank):
        return "structured global nn bank"
    if isinstance(value, StructuredGlobalDensityBank):
        return "structured global density bank"
    if isinstance(value, StructuredLocalMemoryBank):
        return "structured local memory bank"
    if isinstance(value, StructuredGlobalDensityQueryResult):
        return "structured global density query"
    if isinstance(value, StructuredLocalQueryResult):
        return "structured local query"
    return type(value).__name__


def require_normalized_detection_features_payload(
    *,
    detection_features: object,
    stage: str,
) -> MemoryBankPayload:
    """Validate and normalize scorer-owned detection_features payloads."""
    if not is_supported_detection_features_payload(detection_features):
        raise RuntimeError(
            "Anomaly scorer is missing detection_features after "
            f"{stage} or returned an unsupported payload type: "
            f"{describe_payload_family(detection_features)}."
        )
    return normalize_memory_bank_payload(_T.cast(MemoryBankPayload, detection_features))


def validate_fitted_detection_features_payload(
    *,
    expected_memory_bank: MemoryBankPayload,
    actual_detection_features: object,
    stage: str,
) -> MemoryBankPayload:
    """Ensure scorer-owned detection_features preserves the fitted payload family."""
    normalized_actual_detection_features = require_normalized_detection_features_payload(
        detection_features=actual_detection_features,
        stage=stage,
    )
    if isinstance(expected_memory_bank, np.ndarray):
        if not isinstance(normalized_actual_detection_features, np.ndarray):
            raise RuntimeError(
                "Anomaly scorer detection_features family drifted during "
                f"{stage}: expected global ndarray memory bank, got "
                f"{describe_payload_family(normalized_actual_detection_features)}."
            )
    elif isinstance(expected_memory_bank, StructuredGlobalNNBank):
        if not isinstance(
            normalized_actual_detection_features,
            StructuredGlobalNNBank,
        ):
            raise RuntimeError(
                "Anomaly scorer detection_features family drifted during "
                f"{stage}: expected structured global nn bank, got "
                f"{describe_payload_family(normalized_actual_detection_features)}."
            )
    elif isinstance(expected_memory_bank, StructuredGlobalDensityBank):
        if not isinstance(
            normalized_actual_detection_features,
            StructuredGlobalDensityBank,
        ):
            raise RuntimeError(
                "Anomaly scorer detection_features family drifted during "
                f"{stage}: expected structured global density bank, got "
                f"{describe_payload_family(normalized_actual_detection_features)}."
            )
    elif isinstance(expected_memory_bank, StructuredLocalMemoryBank):
        if not isinstance(
            normalized_actual_detection_features,
            StructuredLocalMemoryBank,
        ):
            raise RuntimeError(
                "Anomaly scorer detection_features family drifted during "
                f"{stage}: expected structured local memory bank, got "
                f"{describe_payload_family(normalized_actual_detection_features)}."
            )
    else:
        raise TypeError(
            "Unsupported expected memory-bank payload type during "
            f"{stage}: {describe_payload_family(expected_memory_bank)}."
        )
    return normalized_actual_detection_features


def validate_distance_query_family(
    *,
    memory_bank: MemoryBankPayload,
    distance_query: DistanceQueryPayload,
) -> None:
    """Ensure inference query payload family matches the active bank family."""
    if isinstance(memory_bank, StructuredGlobalDensityBank):
        if not isinstance(distance_query, StructuredGlobalDensityQueryResult):
            raise RuntimeError(
                "Structured global density banks require StructuredGlobalDensityQueryResult "
                f"distance queries; got {describe_payload_family(distance_query)}."
            )
        return
    if isinstance(memory_bank, StructuredLocalMemoryBank):
        if not isinstance(distance_query, StructuredLocalQueryResult):
            raise RuntimeError(
                "Structured local memory banks require StructuredLocalQueryResult "
                f"distance queries; got {describe_payload_family(distance_query)}."
            )
        return
    if isinstance(
        distance_query,
        (StructuredGlobalDensityQueryResult, StructuredLocalQueryResult),
    ):
        raise RuntimeError(
            "Global NN memory banks require tuple distance-query payloads; "
            f"got {describe_payload_family(distance_query)}."
        )


def flatten_structured_global_density_query_payload(
    distance_query: StructuredGlobalDensityQueryResult,
    *,
    component_count: int,
    batchsize: int,
    patch_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flatten structured global density queries to host-global score arrays."""
    expected_vectors = int(batchsize) * int(patch_shape[0]) * int(patch_shape[1])
    patch_scores = np.asarray(distance_query.patch_scores, dtype=np.float32)
    component_ids = np.asarray(distance_query.component_ids, dtype=np.int64)
    component_log_probs = np.asarray(distance_query.component_log_probs, dtype=np.float32)
    component_posteriors = np.asarray(
        distance_query.component_posteriors,
        dtype=np.float32,
    )

    if patch_scores.ndim != 1 or int(patch_scores.shape[0]) != expected_vectors:
        raise RuntimeError(
            "Structured global density query payload must expose 1D patch_scores "
            f"with length {expected_vectors}; got {patch_scores.shape}."
        )
    if component_ids.ndim != 2:
        raise RuntimeError(
            "Structured global density query payload must expose 2D component_ids; "
            f"got {component_ids.shape}."
        )
    if component_ids.shape[0] != expected_vectors or component_ids.shape[1] < 1:
        raise RuntimeError(
            "Structured global density query payload must expose component_ids with "
            f"shape [{expected_vectors}, K>=1]; got {component_ids.shape}."
        )
    if component_log_probs.shape != component_ids.shape:
        raise RuntimeError(
            "Structured global density query payload must align component_log_probs "
            f"with component_ids; got {component_log_probs.shape} and "
            f"{component_ids.shape}."
        )
    if component_posteriors.shape != component_ids.shape:
        raise RuntimeError(
            "Structured global density query payload must align component_posteriors "
            f"with component_ids; got {component_posteriors.shape} and "
            f"{component_ids.shape}."
        )
    if not np.all(np.isfinite(patch_scores)):
        raise RuntimeError(
            "Structured global density query payload requires finite patch_scores."
        )
    if not np.all(np.isfinite(component_log_probs)):
        raise RuntimeError(
            "Structured global density query payload requires finite "
            "component_log_probs."
        )
    if not np.all(np.isfinite(component_posteriors)):
        raise RuntimeError(
            "Structured global density query payload requires finite "
            "component_posteriors."
        )
    if np.any(component_ids < 0):
        raise RuntimeError(
            "Structured global density query payload requires non-negative "
            "component_ids."
        )
    if np.any(component_ids >= int(component_count)):
        raise RuntimeError(
            "Structured global density query payload requires component_ids within "
            f"[0, {int(component_count) - 1}]."
        )
    if np.any(component_posteriors < 0.0) or np.any(component_posteriors > 1.0):
        raise RuntimeError(
            "Structured global density query payload requires component_posteriors "
            "in [0, 1]."
        )

    return (
        patch_scores,
        np.asarray(-component_log_probs, dtype=np.float32),
        component_ids,
    )


def flatten_structured_local_query_payload(
    distance_query: StructuredLocalQueryResult,
    *,
    batchsize: int,
    patch_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flatten row-major structured local query outputs into host-global arrays."""
    expected_patch_shape = (int(patch_shape[0]), int(patch_shape[1]))
    if tuple(distance_query.patch_shape) != expected_patch_shape:
        raise RuntimeError(
            "Structured local query payload patch_shape does not match inference "
            f"patch_shape: query={distance_query.patch_shape} "
            f"expected={expected_patch_shape}."
        )
    if distance_query.flatten_order != "image_major_row_major":
        raise RuntimeError(
            "Structured local query payload must use image_major_row_major "
            f"flatten_order; got {distance_query.flatten_order!r}."
        )

    patch_h, patch_w = expected_patch_shape
    expected_position_count = int(patch_h) * int(patch_w)
    if len(distance_query.position_results) != expected_position_count:
        raise RuntimeError(
            "Structured local query payload is missing aligned positions: "
            f"expected {expected_position_count}, got "
            f"{len(distance_query.position_results)}."
        )

    patch_score_blocks: list[np.ndarray] = []
    query_distance_blocks: list[np.ndarray] = []
    query_nn_blocks: list[np.ndarray] = []
    for entry_index, position_result in enumerate(distance_query.position_results):
        expected_position = (
            entry_index // int(patch_w),
            entry_index % int(patch_w),
        )
        if tuple(position_result.position) != expected_position:
            raise RuntimeError(
                "Structured local query payload must preserve row-major position "
                f"order: expected={expected_position} actual={position_result.position}."
            )
        patch_scores = np.asarray(position_result.patch_scores, dtype=np.float32)
        query_distances = np.asarray(position_result.query_distances, dtype=np.float32)
        query_nns = np.asarray(position_result.query_nns, dtype=np.int64)
        if patch_scores.shape != (int(batchsize),):
            raise RuntimeError(
                "Structured local patch_scores must have shape [batch_size] at "
                f"position={position_result.position}; got {patch_scores.shape}."
            )
        if query_distances.ndim != 2 or query_distances.shape[0] != int(batchsize):
            raise RuntimeError(
                "Structured local query_distances must have shape [batch_size, K] at "
                f"position={position_result.position}; got {query_distances.shape}."
            )
        if query_distances.shape[1] < 1:
            raise RuntimeError(
                "Structured local query_distances must include at least one nearest "
                f"neighbor at position={position_result.position}."
            )
        if query_nns.shape != query_distances.shape:
            raise RuntimeError(
                "Structured local query_nns must match query_distances shape at "
                f"position={position_result.position}; got {query_nns.shape} and "
                f"{query_distances.shape}."
            )
        patch_score_blocks.append(patch_scores)
        query_distance_blocks.append(query_distances)
        query_nn_blocks.append(query_nns)

    flattened_patch_scores = np.stack(patch_score_blocks, axis=1).reshape(
        int(batchsize) * expected_position_count
    )
    flattened_query_distances = np.stack(query_distance_blocks, axis=1).reshape(
        int(batchsize) * expected_position_count,
        -1,
    )
    flattened_query_nns = np.stack(query_nn_blocks, axis=1).reshape(
        int(batchsize) * expected_position_count,
        -1,
    )
    return (
        np.asarray(flattened_patch_scores, dtype=np.float32),
        np.asarray(flattened_query_distances, dtype=np.float32),
        np.asarray(flattened_query_nns, dtype=np.int64),
    )


def flatten_distance_query_payload(
    distance_query: DistanceQueryPayload,
    *,
    memory_bank: MemoryBankPayload,
    batchsize: int,
    patch_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert global or structured local query payloads to score-time arrays."""
    validate_distance_query_family(
        memory_bank=memory_bank,
        distance_query=distance_query,
    )
    if isinstance(distance_query, StructuredGlobalDensityQueryResult):
        if not isinstance(memory_bank, StructuredGlobalDensityBank):
            raise RuntimeError(
                "Structured global density query payload requires StructuredGlobalDensityBank "
                "memory_bank normalization."
            )
        return flatten_structured_global_density_query_payload(
            distance_query,
            component_count=int(np.asarray(memory_bank.component_weights).shape[0]),
            batchsize=batchsize,
            patch_shape=patch_shape,
        )
    if isinstance(distance_query, StructuredLocalQueryResult):
        return flatten_structured_local_query_payload(
            distance_query,
            batchsize=batchsize,
            patch_shape=patch_shape,
        )
    patch_scores, query_distances, query_nns = distance_query
    return (
        np.asarray(patch_scores),
        np.asarray(query_distances),
        np.asarray(query_nns),
    )


def validate_memory_bank_reference_limit(
    memory_bank: MemoryBankPayload,
    *,
    reference_limit: int | None,
    enforce_reference_limit: bool,
    stage: str,
) -> int:
    """Count references and enforce the optional generic host budget contract."""
    reference_count = count_memory_bank_references(memory_bank)
    if reference_limit is None or reference_count <= int(reference_limit):
        return reference_count
    if bool(enforce_reference_limit):
        raise RuntimeError(
            f"{stage} produced more references than configured budget="
            f"{int(reference_limit)} (got {reference_count}). This violates the "
            "bounded-memory streaming contract."
        )
    return reference_count


__all__ = [
    "count_memory_bank_references",
    "describe_payload_family",
    "flatten_distance_query_payload",
    "flatten_structured_global_density_query_payload",
    "flatten_structured_local_query_payload",
    "is_supported_detection_features_payload",
    "normalize_memory_bank_payload",
    "normalize_structured_global_density_bank",
    "normalize_structured_global_nn_bank",
    "resolve_anomaly_scorer_memory_bank",
    "require_normalized_detection_features_payload",
    "validate_anomaly_scorer_memory_bank_compatibility",
    "validate_distance_query_family",
    "validate_fitted_detection_features_payload",
    "validate_memory_bank_reference_limit",
]
