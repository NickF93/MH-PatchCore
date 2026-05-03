"""Plugin-local patch-maker runtime for `patch_align:pc_patchify_align`."""

from __future__ import annotations

import typing as _T

import numpy as _np
import torch as _torch


class PatchMaker:
    """Concrete patch-maker runtime kept local to the patch-align plugin."""

    def __init__(
        self,
        patchsize: int,
        stride: _T.Optional[int] = None,
    ) -> None:
        self.patchsize: int = patchsize
        self.stride: int = stride if stride is not None else patchsize

    def patchify(
        self,
        features: _torch.Tensor,
        return_spatial_info: bool = False,
    ) -> _T.Union[_torch.Tensor, _T.Tuple[_torch.Tensor, _T.List[int]]]:
        padding = int((self.patchsize - 1) / 2)
        unfolder = _torch.nn.Unfold(
            kernel_size=self.patchsize,
            stride=self.stride,
            padding=padding,
            dilation=1,
        )
        unfolded_features: _torch.Tensor = unfolder(features)
        number_of_total_patches = []
        for spatial_dim in features.shape[-2:]:
            n_patches = (
                spatial_dim + 2 * padding - 1 * (self.patchsize - 1) - 1
            ) / self.stride + 1
            number_of_total_patches.append(int(n_patches))
        unfolded_features = unfolded_features.reshape(
            *features.shape[:2],
            self.patchsize,
            self.patchsize,
            -1,
        )
        unfolded_features = unfolded_features.permute(0, 4, 1, 2, 3)

        if return_spatial_info:
            return unfolded_features, number_of_total_patches
        return unfolded_features

    def unpatch_scores(
        self,
        x: _torch.Tensor,
        batchsize: int,
    ) -> _torch.Tensor:
        return x.reshape(batchsize, -1, *x.shape[1:])

    def score(
        self,
        x: _torch.Tensor | _np.ndarray,
    ) -> _torch.Tensor | _np.ndarray:
        was_numpy = False
        if isinstance(x, _np.ndarray):
            was_numpy = True
            x = _torch.from_numpy(x)
        while x.ndim > 1:
            x = _torch.max(x, dim=-1).values
        if was_numpy:
            return x.numpy()
        return x
