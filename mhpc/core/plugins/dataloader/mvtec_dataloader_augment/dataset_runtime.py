"""Plugin-local MVTec dataset and dataloader runtime."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import tv_tensors  # type: ignore[import-untyped]

from .augmentation_runtime import (
    AUGMENTATION_MODES,
    JointAugmentor,
    TransformLike,
    apply_joint_transform as _apply_joint_transform_impl,
    build_transforms,
)

__all__ = [
    "AUGMENTATION_MODES",
    "JointAugmentor",
    "build_transforms",
    "apply_joint_transform",
    "MVTecDataset",
    "create_dataloaders",
]


def apply_joint_transform(
    transform: TransformLike,
    image: Tensor,
    mask: tv_tensors.Mask | Tensor,
    deterministic_seed: int | None = None,
) -> tuple[Tensor, Tensor]:
    """Backward-compatible re-export for joint image/mask transform call."""
    return _apply_joint_transform_impl(
        transform=transform,
        image=image,
        mask=mask,
        deterministic_seed=deterministic_seed,
    )


class MVTecDataset(Dataset):
    """MVTec AD segmentation dataset using torchvision.transforms.v2."""

    def __init__(
        self,
        root: str,
        category: str,
        split: str,
        augment_cfg: dict[str, Any] | None = None,
        augment: bool = False,
        augment_mode: str = "independent",
        augment_seed: int | None = None,
        augment_seed_devices: tuple[int, ...] | None = None,
        dtype: torch.dtype = torch.float32,
        resize: tuple[int, int] = (256, 256),
        img_size: tuple[int, int] = (256, 256),
    ) -> None:
        if augment_mode not in AUGMENTATION_MODES:
            raise ValueError(
                "augment_mode must be one of: "
                f"{', '.join(sorted(AUGMENTATION_MODES))}"
            )

        self.root = Path(root) / category
        self.split = split
        self.dtype = dtype
        self._augment_mode = "none"
        if augment and split == "train":
            self._augment_mode = augment_mode
        self._augment_seed = augment_seed
        if self._augment_mode == "pass_consistent" and augment_seed is None:
            raise ValueError(
                "augment_seed is required when augment_mode='pass_consistent'"
            )

        self.img_paths: list[Path] = sorted((self.root / split).rglob("*.png"))
        enable_random_augment = self._augment_mode in {
            "independent",
            "pass_consistent",
        }
        self.transform = build_transforms(
            resize=resize,
            img_size=img_size,
            augment=enable_random_augment,
            cfg=augment_cfg,
            dtype=dtype,
        )
        self._augmentor = JointAugmentor(
            transform=self.transform,
            deterministic_seed_devices=augment_seed_devices,
        )

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        img_path = self.img_paths[idx]
        image_bgr = cv2.imread(str(img_path))
        if image_bgr is None:
            raise FileNotFoundError(f"Failed to read image file: {img_path}")
        img_np = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        height, width = img_np.shape[:2]

        rel = img_path.relative_to(self.root / self.split)
        if rel.parts[0] == "good":
            mask_np = np.zeros((height, width), dtype=np.uint8)
        else:
            defect = rel.parts[0]
            mask_name = f"{img_path.stem}_mask.png"
            gt_path = self.root / "ground_truth" / defect / mask_name
            if gt_path.exists():
                loaded_mask = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
                if loaded_mask is None:
                    raise FileNotFoundError(f"Failed to read mask file: {gt_path}")
                mask_np = loaded_mask
            else:
                mask_np = np.zeros((height, width), dtype=np.uint8)

        mask_tt = tv_tensors.Mask(mask_np)
        img_tt = torch.from_numpy(img_np).permute(2, 0, 1).contiguous()

        deterministic_seed: int | None = None
        if self._augment_mode == "pass_consistent":
            if self._augment_seed is None:
                raise RuntimeError(
                    "augment_seed must be set for pass_consistent augmentation mode"
                )
            deterministic_seed = int(self._augment_seed + idx)

        img_transformed, mask_transformed_opt = self._augmentor(
            image=img_tt,
            mask=mask_tt,
            deterministic_seed=deterministic_seed,
        )
        if mask_transformed_opt is None:
            raise RuntimeError(
                "Joint augmentor returned no mask for image+mask dataset sample."
            )

        mask_bin = (mask_transformed_opt > 0).to(dtype=self.dtype)
        img_out = img_transformed.to(dtype=self.dtype)
        return img_out, mask_bin


def create_dataloaders(
    root: str,
    category: str,
    batch_size: int = 32,
    num_workers: int = 4,
    augment_cfg: dict[str, Any] | None = None,
    augment: bool = False,
    augment_mode: str = "independent",
    augment_seed: int | None = None,
    augment_seed_devices: tuple[int, ...] | None = None,
    dtype: torch.dtype = torch.float32,
    img_size: tuple[int, int] = (256, 256),
) -> tuple[DataLoader, DataLoader]:
    """Create train and test DataLoaders for one MVTec category."""
    train_dataset = MVTecDataset(
        root=root,
        category=category,
        split="train",
        augment_cfg=augment_cfg,
        augment=augment,
        augment_mode=augment_mode,
        augment_seed=augment_seed,
        augment_seed_devices=augment_seed_devices,
        dtype=dtype,
        img_size=img_size,
    )
    test_dataset = MVTecDataset(
        root=root,
        category=category,
        split="test",
        augment_cfg=None,
        augment=False,
        augment_mode="none",
        augment_seed=None,
        augment_seed_devices=augment_seed_devices,
        dtype=dtype,
        img_size=img_size,
    )
    persistent_workers = bool(num_workers > 0)
    multiprocessing_kwargs: dict[str, Any] = {}
    if num_workers > 0:
        multiprocessing_kwargs["multiprocessing_context"] = "spawn"

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=persistent_workers,
        **multiprocessing_kwargs,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent_workers,
        **multiprocessing_kwargs,
    )

    return train_loader, test_loader
