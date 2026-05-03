"""Plugin-local parameter binding/parsing for `mvtec_dataloader_augment`."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config_primitives import (
    _require_bool,
    _require_int,
    _require_int_pair,
    _require_non_empty_string,
    _require_optional_bool,
    _require_optional_float,
    _require_optional_int_list,
    _require_optional_mapping,
    _require_optional_string,
)
from ..contracts import DataLoaderBindContextLike

_ALLOWED_KEYS = frozenset(
    {
        "dataset_root",
        "categories",
        "batch_size",
        "num_workers",
        "img_size",
        "train_augment_enabled",
        "streaming_augmentation_policy",
        "train_augment_seed_devices",
        "train_augment_cfg",
        "train_augment_overrides",
    }
)


@dataclass(frozen=True)
class MVTecDataLoaderParams:
    """Canonical params payload bound to one dataloader plugin instance."""

    dataset_root: Path
    categories: tuple[str, ...]
    batch_size: int
    num_workers: int
    img_size: tuple[int, int]
    train_augment_enabled: bool
    streaming_augmentation_policy: str
    train_augment_seed_devices: tuple[int, ...]
    train_augment_cfg: dict[str, Any]
    train_augment_overrides: dict[str, dict[str, Any]]


def _normalize_training_contract(value: object) -> str:
    raw_value = value.name if hasattr(value, "name") else value
    if not isinstance(raw_value, str):
        raise TypeError(
            "training_contract must be a string token: "
            f"type={type(raw_value).__name__}"
        )
    normalized = raw_value.strip().upper()
    if normalized not in {"OFFLINE", "STREAMING"}:
        raise ValueError(
            "training_contract must be one of {'OFFLINE', 'STREAMING'}: "
            f"value={raw_value!r}"
        )
    return normalized


def _ensure_allowed_keys(
    mapping: Mapping[str, Any],
    *,
    allowed_keys: set[str] | frozenset[str],
    context: str,
) -> None:
    unknown_keys = sorted(str(key) for key in set(mapping.keys()) - set(allowed_keys))
    if unknown_keys:
        raise ValueError(
            f"{context} contains unsupported keys: {', '.join(unknown_keys)}"
        )


def _validate_probability(*, probability: float, context: str) -> None:
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"{context} must be in [0, 1]")


def _parse_train_augment_flip_section(
    cfg: Mapping[str, Any],
    *,
    section_name: str,
    context: str,
    include_defaults: bool,
) -> dict[str, Any] | None:
    section = _require_optional_mapping(cfg, section_name) or {}
    if not section:
        if include_defaults:
            return {"enable": False, "p": 0.5}
        return None
    _ensure_allowed_keys(
        section,
        allowed_keys={"enable", "p"},
        context=f"{context}.{section_name}",
    )
    parsed: dict[str, Any] = {}
    if include_defaults or "enable" in section:
        parsed["enable"] = _require_optional_bool(section, "enable", default=False)
    if include_defaults or "p" in section:
        prob = _require_optional_float(section, "p", default=0.5)
        _validate_probability(
            probability=prob,
            context=f"{context}.{section_name}.p",
        )
        parsed["p"] = prob
    return parsed


def _parse_train_augment_rotate_section(
    cfg: Mapping[str, Any],
    *,
    context: str,
    include_defaults: bool,
) -> dict[str, Any] | None:
    section_name = "rotate"
    section = _require_optional_mapping(cfg, section_name) or {}
    if not section:
        if include_defaults:
            return {"enable": False, "limit": 30.0, "p": 0.5}
        return None
    _ensure_allowed_keys(
        section,
        allowed_keys={"enable", "limit", "p"},
        context=f"{context}.{section_name}",
    )
    parsed: dict[str, Any] = {}
    if include_defaults or "enable" in section:
        parsed["enable"] = _require_optional_bool(section, "enable", default=False)
    if include_defaults or "limit" in section:
        limit = _require_optional_float(section, "limit", default=30.0)
        if limit < 0.0:
            raise ValueError(f"{context}.rotate.limit must be >= 0")
        parsed["limit"] = limit
    if include_defaults or "p" in section:
        prob = _require_optional_float(section, "p", default=0.5)
        _validate_probability(
            probability=prob,
            context=f"{context}.rotate.p",
        )
        parsed["p"] = prob
    return parsed


def _parse_train_augment_random90_section(
    cfg: Mapping[str, Any],
    *,
    context: str,
    include_defaults: bool,
) -> dict[str, Any] | None:
    section_name = "random90"
    section = _require_optional_mapping(cfg, section_name) or {}
    if not section:
        if include_defaults:
            return {"enable": False}
        return None
    _ensure_allowed_keys(
        section,
        allowed_keys={"enable"},
        context=f"{context}.{section_name}",
    )
    return {"enable": _require_optional_bool(section, "enable", default=False)}


def _parse_train_augment_color_jitter_section(
    cfg: Mapping[str, Any],
    *,
    context: str,
    include_defaults: bool,
) -> dict[str, Any] | None:
    section_name = "color_jitter"
    section = _require_optional_mapping(cfg, section_name) or {}
    if not section:
        if include_defaults:
            return {
                "enable": False,
                "p": 0.5,
                "brightness": 0.2,
                "contrast": 0.2,
                "saturation": 0.2,
                "hue": 0.1,
            }
        return None
    _ensure_allowed_keys(
        section,
        allowed_keys={
            "enable",
            "p",
            "brightness",
            "contrast",
            "saturation",
            "hue",
        },
        context=f"{context}.{section_name}",
    )
    parsed: dict[str, Any] = {}
    if include_defaults or "enable" in section:
        parsed["enable"] = _require_optional_bool(section, "enable", default=False)
    if include_defaults or "p" in section:
        prob = _require_optional_float(section, "p", default=0.5)
        _validate_probability(
            probability=prob,
            context=f"{context}.color_jitter.p",
        )
        parsed["p"] = prob
    if include_defaults or "brightness" in section:
        brightness = _require_optional_float(section, "brightness", default=0.2)
        if brightness < 0.0:
            raise ValueError(f"{context}.color_jitter.brightness must be >= 0")
        parsed["brightness"] = brightness
    if include_defaults or "contrast" in section:
        contrast = _require_optional_float(section, "contrast", default=0.2)
        if contrast < 0.0:
            raise ValueError(f"{context}.color_jitter.contrast must be >= 0")
        parsed["contrast"] = contrast
    if include_defaults or "saturation" in section:
        saturation = _require_optional_float(section, "saturation", default=0.2)
        if saturation < 0.0:
            raise ValueError(f"{context}.color_jitter.saturation must be >= 0")
        parsed["saturation"] = saturation
    if include_defaults or "hue" in section:
        hue = _require_optional_float(section, "hue", default=0.1)
        if not 0.0 <= hue <= 0.5:
            raise ValueError(f"{context}.color_jitter.hue must be in [0, 0.5]")
        parsed["hue"] = hue
    return parsed


def _parse_train_augment_cfg(
    cfg: Mapping[str, Any],
    *,
    context: str,
    include_defaults: bool,
) -> dict[str, Any]:
    allowed_sections = {"hflip", "vflip", "rotate", "random90", "color_jitter"}
    _ensure_allowed_keys(cfg, allowed_keys=allowed_sections, context=context)

    parsed: dict[str, Any] = {}
    hflip = _parse_train_augment_flip_section(
        cfg,
        section_name="hflip",
        context=context,
        include_defaults=include_defaults,
    )
    if hflip is not None:
        parsed["hflip"] = hflip
    vflip = _parse_train_augment_flip_section(
        cfg,
        section_name="vflip",
        context=context,
        include_defaults=include_defaults,
    )
    if vflip is not None:
        parsed["vflip"] = vflip
    rotate = _parse_train_augment_rotate_section(
        cfg,
        context=context,
        include_defaults=include_defaults,
    )
    if rotate is not None:
        parsed["rotate"] = rotate
    random90 = _parse_train_augment_random90_section(
        cfg,
        context=context,
        include_defaults=include_defaults,
    )
    if random90 is not None:
        parsed["random90"] = random90
    color_jitter = _parse_train_augment_color_jitter_section(
        cfg,
        context=context,
        include_defaults=include_defaults,
    )
    if color_jitter is not None:
        parsed["color_jitter"] = color_jitter
    return parsed


def _parse_train_augment_overrides(
    cfg: Mapping[str, Any],
    *,
    categories: tuple[str, ...],
) -> dict[str, dict[str, Any]]:
    allowed_categories = set(categories)
    unknown_categories: list[str] = []
    parsed: dict[str, dict[str, Any]] = {}

    for raw_category, raw_override in cfg.items():
        if not isinstance(raw_category, str) or not raw_category.strip():
            raise ValueError(
                "pipeline.slots.dataloader.params.train_augment_overrides keys "
                "must be non-empty strings"
            )
        category = raw_category.strip()
        if category not in allowed_categories:
            unknown_categories.append(category)
            continue
        if not isinstance(raw_override, Mapping):
            raise ValueError(
                "pipeline.slots.dataloader.params.train_augment_overrides."
                f"{category} must be a mapping"
            )
        parsed[category] = _parse_train_augment_cfg(
            raw_override,
            context=(
                "pipeline.slots.dataloader.params.train_augment_overrides."
                f"{category}"
            ),
            include_defaults=False,
        )

    if unknown_categories:
        raise ValueError(
            "pipeline.slots.dataloader.params.train_augment_overrides contains "
            f"unknown categories: {', '.join(sorted(set(unknown_categories)))}"
        )
    return parsed


def _parse_mvtec_dataloader_params(
    params: Mapping[str, Any],
    *,
    repo_root: Path,
) -> MVTecDataLoaderParams:
    context = "pipeline.slots.dataloader.params"
    _ensure_allowed_keys(params, allowed_keys=_ALLOWED_KEYS, context=context)

    dataset_root = _resolve_repo_relative_dataset_root(
        _require_non_empty_string(params, "dataset_root"),
        repo_root=repo_root,
    )

    categories_obj = params.get("categories")
    if not isinstance(categories_obj, list) or not categories_obj:
        raise ValueError(f"{context}.categories must be a non-empty list of strings")
    categories = tuple(_require_non_empty_string({"value": item}, "value") for item in categories_obj)

    batch_size = _require_int(params, "batch_size")
    if batch_size <= 0:
        raise ValueError(f"{context}.batch_size must be > 0")
    num_workers = _require_int(params, "num_workers")
    if num_workers < 0:
        raise ValueError(f"{context}.num_workers must be >= 0")
    img_size = _require_int_pair(params, "img_size")

    train_augment_enabled = _require_bool(params, "train_augment_enabled")
    streaming_augmentation_policy = _require_optional_string(
        params,
        "streaming_augmentation_policy",
        default="independent",
    )
    allowed_streaming_policies = {"none", "independent", "pass_consistent"}
    if streaming_augmentation_policy not in allowed_streaming_policies:
        raise ValueError(
            f"{context}.streaming_augmentation_policy must be one of: "
            f"{', '.join(sorted(allowed_streaming_policies))}"
        )

    train_augment_seed_devices = _require_optional_int_list(
        params,
        "train_augment_seed_devices",
        default=(),
    )
    for device_idx, value in enumerate(train_augment_seed_devices):
        if value < 0:
            raise ValueError(
                f"{context}.train_augment_seed_devices values must be >= 0; "
                f"got index={device_idx} value={value}"
            )

    train_augment_cfg_obj = params.get("train_augment_cfg", {})
    if not isinstance(train_augment_cfg_obj, Mapping):
        raise ValueError(f"{context}.train_augment_cfg must be a mapping")
    train_augment_cfg = _parse_train_augment_cfg(
        train_augment_cfg_obj,
        context=f"{context}.train_augment_cfg",
        include_defaults=True,
    )

    train_augment_overrides_obj = params.get("train_augment_overrides", {})
    if not isinstance(train_augment_overrides_obj, Mapping):
        raise ValueError(f"{context}.train_augment_overrides must be a mapping")
    train_augment_overrides = _parse_train_augment_overrides(
        train_augment_overrides_obj,
        categories=categories,
    )

    return MVTecDataLoaderParams(
        dataset_root=dataset_root,
        categories=categories,
        batch_size=batch_size,
        num_workers=num_workers,
        img_size=img_size,
        train_augment_enabled=train_augment_enabled,
        streaming_augmentation_policy=streaming_augmentation_policy,
        train_augment_seed_devices=tuple(train_augment_seed_devices),
        train_augment_cfg=train_augment_cfg,
        train_augment_overrides=train_augment_overrides,
    )


def _resolve_repo_relative_dataset_root(raw_value: str, *, repo_root: Path) -> Path:
    authored_path = Path(raw_value).expanduser()
    if authored_path.is_absolute():
        return authored_path
    return repo_root / authored_path


class MVTecDataLoaderParamBindingMixin:
    """Dataloader slot mixin with plugin-local parameter parsing."""

    _bound_params: dict[str, Any]
    _bound_bind_context: DataLoaderBindContextLike
    _dataloader_params: MVTecDataLoaderParams

    def bind_params(
        self,
        *,
        params: Mapping[str, Any],
        bind_context: DataLoaderBindContextLike,
    ) -> None:
        if not isinstance(params, Mapping):
            raise TypeError(
                "params must be a mapping for plugin bind_params: "
                f"type={type(params).__name__}"
            )
        _normalize_training_contract(getattr(bind_context, "training_contract", None))
        seed = getattr(bind_context, "seed", None)
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError(
                "bind_context.seed must be an integer: "
                f"type={type(seed).__name__}"
            )
        repo_root = getattr(bind_context, "repo_root", None)
        if not isinstance(repo_root, Path):
            raise TypeError(
                "bind_context.repo_root must be a pathlib.Path: "
                f"type={type(repo_root).__name__}"
            )
        self._bound_params = dict(params)
        self._bound_bind_context = bind_context
        self._dataloader_params = _parse_mvtec_dataloader_params(
            self._bound_params,
            repo_root=repo_root,
        )

    def resolve_dataset_plan(self) -> tuple[str, ...]:
        return tuple(self._dataloader_params.categories)
