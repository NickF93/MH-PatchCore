"""Plugin-local joint image/mask augmentation runtime utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

import torch
from torch import Tensor
from torchvision import tv_tensors  # type: ignore[import-untyped]
from torchvision.transforms import v2  # type: ignore[import-untyped]

AUGMENTATION_MODES = frozenset({"none", "independent", "pass_consistent"})
__all__ = [
    "AUGMENTATION_MODES",
    "TransformLike",
    "JointAugmentor",
    "build_transforms",
    "apply_joint_transform",
]


class TransformLike(Protocol):
    """Protocol for callable transforms used by ``JointAugmentor``."""

    def __call__(
        self,
        image: Tensor,
        mask: Tensor | None = None,
    ) -> Any:
        """Apply transform to image-only or image+mask payloads."""


@dataclass(frozen=True)
class JointAugmentor:
    """Apply a torchvision v2 transform to image-only or image+mask inputs."""

    transform: TransformLike
    deterministic_seed_devices: Sequence[int] | None = None

    def __post_init__(self) -> None:
        if self.deterministic_seed_devices is None:
            return
        for device_idx in self.deterministic_seed_devices:
            if not isinstance(device_idx, int):
                raise TypeError(
                    "deterministic_seed_devices entries must be integers; "
                    f"got {type(device_idx).__name__}."
                )
            if device_idx < 0:
                raise ValueError(
                    "deterministic_seed_devices entries must be >= 0; "
                    f"got {device_idx}."
                )

    def __call__(
        self,
        image: Tensor,
        mask: tv_tensors.Mask | Tensor | None = None,
        deterministic_seed: int | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        """Apply transform with optional deterministic replay."""
        if not isinstance(image, torch.Tensor):
            raise TypeError(
                "image must be a torch.Tensor for JointAugmentor; "
                f"got {type(image).__name__}."
            )
        if image.ndim not in {3, 4}:
            raise ValueError(
                "image must have shape [C,H,W] or [B,C,H,W] for JointAugmentor; "
                f"got ndim={image.ndim}."
            )
        mask_normalized = _normalize_mask(mask)

        if deterministic_seed is None:
            if mask_normalized is None:
                transformed_image = self.transform(image)
                return transformed_image, None
            transformed_image, transformed_mask = self.transform(image, mask_normalized)
            return transformed_image, transformed_mask

        devices = list(self.deterministic_seed_devices or [])
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(int(deterministic_seed))
            if mask_normalized is None:
                transformed_image = self.transform(image)
                return transformed_image, None
            transformed_image, transformed_mask = self.transform(
                image,
                mask_normalized,
            )
            return transformed_image, transformed_mask


def build_transforms(
    resize: tuple[int, int],
    img_size: tuple[int, int],
    augment: bool,
    cfg: Mapping[str, Any] | None,
    dtype: torch.dtype,
) -> v2.Compose:
    """Build torchvision v2 transform chain for train/test processing."""
    _validate_hw_size("resize", resize)
    _validate_hw_size("img_size", img_size)
    ops: list[Any] = []
    aug_cfg = _normalize_augment_cfg(cfg)

    if augment and aug_cfg:
        if _is_enabled(aug_cfg, "hflip"):
            ops.append(v2.RandomHorizontalFlip(p=_get_prob(aug_cfg, "hflip")))

        if _is_enabled(aug_cfg, "vflip"):
            ops.append(v2.RandomVerticalFlip(p=_get_prob(aug_cfg, "vflip")))

        if _is_enabled(aug_cfg, "rotate"):
            limit = float(_get_nested(aug_cfg, "rotate", "limit", default=30.0))
            if limit < 0.0:
                raise ValueError(
                    f"augment cfg value 'rotate.limit' must be >= 0, got {limit}."
                )
            p_rotate = _get_prob(aug_cfg, "rotate")
            rotate_module = torch.nn.ModuleList(
                [
                    v2.RandomRotation(
                        degrees=limit,
                        interpolation=v2.InterpolationMode.BILINEAR,
                        expand=False,
                        center=None,
                        fill=0,
                    )
                ]
            )
            ops.append(v2.RandomApply(rotate_module, p=p_rotate))

        if _is_enabled(aug_cfg, "random90"):
            identity = v2.Lambda(lambda x: x)
            random90_modules = [
                identity,
                v2.RandomRotation(
                    degrees=(90, 90),
                    interpolation=v2.InterpolationMode.BILINEAR,
                ),
                v2.RandomRotation(
                    degrees=(180, 180),
                    interpolation=v2.InterpolationMode.BILINEAR,
                ),
                v2.RandomRotation(
                    degrees=(270, 270),
                    interpolation=v2.InterpolationMode.BILINEAR,
                ),
            ]
            ops.append(v2.RandomChoice(random90_modules, p=[0.25, 0.25, 0.25, 0.25]))

        if _is_enabled(aug_cfg, "color_jitter"):
            color_jitter_cfg = _require_submapping(aug_cfg, "color_jitter")
            p_color = _get_prob(aug_cfg, "color_jitter")
            color_module = torch.nn.ModuleList(
                [
                    v2.ColorJitter(
                        brightness=float(color_jitter_cfg.get("brightness", 0.2)),
                        contrast=float(color_jitter_cfg.get("contrast", 0.2)),
                        saturation=float(color_jitter_cfg.get("saturation", 0.2)),
                        hue=float(color_jitter_cfg.get("hue", 0.1)),
                    )
                ]
            )
            ops.append(v2.RandomApply(color_module, p=p_color))

    ops.extend(
        [
            v2.Resize(
                size=resize,
                antialias=True,
                interpolation=v2.InterpolationMode.BILINEAR,
            ),
            v2.CenterCrop(size=img_size),
            v2.ToDtype(dtype, scale=True),
            v2.ToImage(),
            v2.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    return v2.Compose(ops)


def apply_joint_transform(
    transform: TransformLike,
    image: Tensor,
    mask: tv_tensors.Mask | Tensor,
    deterministic_seed: int | None = None,
) -> tuple[Tensor, Tensor]:
    """Backward-compatible helper for image+mask transform calls."""
    augmentor = JointAugmentor(transform=transform)
    image_out, mask_out = augmentor(
        image=image,
        mask=mask,
        deterministic_seed=deterministic_seed,
    )
    if mask_out is None:
        raise RuntimeError(
            "Joint transform unexpectedly returned no mask for image+mask input."
        )
    return image_out, mask_out


def _normalize_augment_cfg(cfg: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if cfg is None:
        return {}
    if not isinstance(cfg, Mapping):
        raise TypeError(
            "augment cfg must be a mapping or None; "
            f"got {type(cfg).__name__}."
        )
    return cfg


def _require_submapping(cfg: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = cfg.get(key, {})
    if not isinstance(value, Mapping):
        raise TypeError(f"augment cfg section '{key}' must be a mapping.")
    return value


def _is_enabled(cfg: Mapping[str, Any], key: str) -> bool:
    section = _require_submapping(cfg, key)
    return bool(section.get("enable", False))


def _get_prob(cfg: Mapping[str, Any], key: str) -> float:
    section = _require_submapping(cfg, key)
    prob = float(section.get("p", 0.5))
    if prob < 0.0 or prob > 1.0:
        raise ValueError(
            f"augment cfg probability '{key}.p' must be in [0, 1], got {prob}."
        )
    return prob


def _get_nested(
    cfg: Mapping[str, Any],
    section_key: str,
    field_key: str,
    *,
    default: float,
) -> float:
    section = _require_submapping(cfg, section_key)
    return float(section.get(field_key, default))


def _normalize_mask(mask: tv_tensors.Mask | Tensor | None) -> tv_tensors.Mask | None:
    if mask is None:
        return None
    if not isinstance(mask, torch.Tensor):
        raise TypeError(
            "mask must be a torch.Tensor, tv_tensors.Mask, or None; "
            f"got {type(mask).__name__}."
        )
    if mask.ndim not in {2, 3}:
        raise ValueError(
            "mask must have shape [H,W] or [B,H,W] for JointAugmentor; "
            f"got ndim={mask.ndim}."
        )
    if isinstance(mask, tv_tensors.Mask):
        return mask
    return tv_tensors.Mask(mask)


def _validate_hw_size(name: str, value: tuple[int, int]) -> None:
    if len(value) != 2:
        raise ValueError(f"{name} must contain exactly 2 integers.")
    if any(int(dim) <= 0 for dim in value):
        raise ValueError(
            f"{name} dimensions must be positive integers, got {value}."
        )
