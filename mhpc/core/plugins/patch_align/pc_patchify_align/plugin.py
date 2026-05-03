"""PatchCore patchify+align plugin implementation."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from ..contracts import PatchAlignPlugin
from .param_binding import PCPatchifyAlignParamBindingMixin


class PCPatchifyAlignPlugin(PCPatchifyAlignParamBindingMixin, PatchAlignPlugin):
    """Behavior-preserving plugin for patchify+align logic."""

    supports_streaming: bool = True
    requires_full_dataset: bool = False

    def create_patch_maker(self) -> Any:
        return self.build_patch_maker()

    def patchify_and_align(
        self,
        *,
        features: list[torch.Tensor],
        patch_maker: Any,
    ) -> tuple[list[torch.Tensor], list[list[int]]]:
        patched = [patch_maker.patchify(x, return_spatial_info=True) for x in features]
        patch_shapes = [x[1] for x in patched]
        aligned_features = [x[0] for x in patched]
        ref_num_patches = patch_shapes[0]

        for i in range(1, len(aligned_features)):
            layer_features = aligned_features[i]
            patch_dims = patch_shapes[i]

            layer_features = layer_features.reshape(
                layer_features.shape[0],
                int(patch_dims[0]),
                int(patch_dims[1]),
                *layer_features.shape[2:],
            )
            layer_features = layer_features.permute(0, -3, -2, -1, 1, 2)
            perm_base_shape = layer_features.shape
            layer_features = layer_features.reshape(-1, *layer_features.shape[-2:])
            layer_features = F.interpolate(
                layer_features.unsqueeze(1),
                size=(int(ref_num_patches[0]), int(ref_num_patches[1])),
                mode="bilinear",
                align_corners=False,
            )
            layer_features = layer_features.squeeze(1)
            layer_features = layer_features.reshape(
                *perm_base_shape[:-2],
                int(ref_num_patches[0]),
                int(ref_num_patches[1]),
            )
            layer_features = layer_features.permute(0, -2, -1, 1, 2, 3)
            layer_features = layer_features.reshape(
                len(layer_features),
                -1,
                *layer_features.shape[-3:],
            )
            aligned_features[i] = layer_features

        aligned_features = [x.reshape(-1, *x.shape[-3:]) for x in aligned_features]
        return aligned_features, patch_shapes
