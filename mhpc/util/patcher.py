'''
DISCLAIMER: This code is primarily based on the following source:
https://github.com/amazon-science/patchcore-inspection
Specifically, it has been adapted from the file:
https://github.com/amazon-science/patchcore-inspection/blob/main/src/patchcore/patchcore.py
'''

import torch as _torch
import typing as _T
import numpy as _np

class PatchMaker:
    def __init__(
        self,
        patchsize: int,
        stride: _T.Optional[int] = None
    ):
        self.patchsize: int = patchsize
        self.stride: int = stride if stride is not None else patchsize

    def patchify(
        self,
        features: _torch.Tensor,
        return_spatial_info: bool = False
    ) -> _T.Union[_torch.Tensor, _T.Tuple[_torch.Tensor, _T.List[int]]]:
        """Convert a tensor into a tensor of respective patches.
        Args:
            x: [torch.Tensor, bs x c x w x h]
        Returns:
            x: [torch.Tensor, bs * w//stride * h//stride, c, patchsize,
            patchsize]
        """
        padding = int((self.patchsize - 1) / 2)
        unfolder = _torch.nn.Unfold(
            kernel_size=self.patchsize, stride=self.stride, padding=padding, dilation=1
        )
        unfolded_features: _torch.Tensor = unfolder(features)
        number_of_total_patches = []
        for s in features.shape[-2:]:
            n_patches = (
                s + 2 * padding - 1 * (self.patchsize - 1) - 1
            ) / self.stride + 1
            number_of_total_patches.append(int(n_patches))
        unfolded_features = unfolded_features.reshape(
            *features.shape[:2], self.patchsize, self.patchsize, -1
        )
        unfolded_features = unfolded_features.permute(0, 4, 1, 2, 3)

        if return_spatial_info:
            return unfolded_features, number_of_total_patches
        return unfolded_features

    def unpatch_scores(
            self,
            x: _torch.Tensor,
            batchsize: int
        ) -> _torch.Tensor:
        return x.reshape(batchsize, -1, *x.shape[1:])

    def score(
        self,
        x: _T.Union[_torch.Tensor, _np.ndarray],
    ) -> _T.Union[_np.ndarray, _torch.Tensor]:
        was_numpy = False
        if isinstance(x, _np.ndarray):
            was_numpy = True
            x = _torch.from_numpy(x)
        while x.ndim > 1:
            x = _torch.max(x, dim=-1).values
        if was_numpy:
            return x.numpy()
        return x
