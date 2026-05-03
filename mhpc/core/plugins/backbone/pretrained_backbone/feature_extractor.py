"""Feature extractor helpers local to the unified pretrained backbone plugin."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

import torch


class _ExtractorStrategy(Protocol):
    def extract(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        ...

    def close(self) -> None:
        ...


class _SequentialChildrenStrategy:
    """Run direct child modules sequentially and stop at the last requested one."""

    def __init__(self, backbone: torch.nn.Module, layers_to_extract_from: list[str]) -> None:
        self._backbone = backbone
        self._layers = list(layers_to_extract_from)
        self._children: list[tuple[str, torch.nn.Module]] = list(backbone.named_children())
        child_names = {name for name, _ in self._children}
        missing = [layer for layer in self._layers if layer not in child_names]
        if missing:
            raise ValueError(
                "Sequential extraction requires layer names to match direct "
                f"children. missing={missing}"
            )

    def extract(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs: dict[str, torch.Tensor] = {}
        current = x
        for name, module in self._children:
            current = module(current)
            if name in self._layers:
                outputs[name] = current
            if len(outputs) == len(self._layers):
                break

        missing_layers = [layer for layer in self._layers if layer not in outputs]
        if missing_layers:
            raise RuntimeError(
                "Sequential feature extraction failed; missing outputs for layers: "
                f"{missing_layers}"
            )
        return {layer: outputs[layer] for layer in self._layers}

    def close(self) -> None:
        return


class _HookBasedStrategy:
    """Generic hook-based extraction for non-sequential backbone graphs."""

    def __init__(self, backbone: torch.nn.Module, layers_to_extract_from: list[str]) -> None:
        self._backbone = backbone
        self._layers = list(layers_to_extract_from)
        self._outputs: dict[str, torch.Tensor] = {}
        self._hooks: list[Any] = []
        self._forward_fn = self._resolve_forward_fn()
        self._register_hooks()

    def _resolve_forward_fn(self) -> Callable[[torch.Tensor], torch.Tensor | Any]:
        forward_features = getattr(self._backbone, "forward_features", None)
        if callable(forward_features):
            return forward_features
        return self._backbone

    def _register_hooks(self) -> None:
        available_layers = dict(self._backbone.named_modules())
        missing_layers = [
            layer for layer in self._layers if layer not in available_layers
        ]
        if missing_layers:
            raise ValueError(
                "Layer(s) not found in backbone. "
                f"missing={missing_layers} available={list(available_layers.keys())}"
            )

        for layer_name in self._layers:
            module = available_layers[layer_name]
            self._hooks.append(module.register_forward_hook(self._build_hook(layer_name)))

    def _build_hook(self, layer_name: str):
        def hook(
            _module: torch.nn.Module,
            _inputs: tuple[torch.Tensor, ...],
            output: torch.Tensor,
        ) -> None:
            self._outputs[layer_name] = output

        return hook

    def extract(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        self._outputs = {}
        _ = self._forward_fn(x)
        missing_layers = [layer for layer in self._layers if layer not in self._outputs]
        if missing_layers:
            raise RuntimeError(
                "Feature extraction failed; missing outputs for layers: "
                f"{missing_layers}"
            )
        return {layer: self._outputs[layer] for layer in self._layers}

    def close(self) -> None:
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()


class FeatureExtractor(torch.nn.Module):
    """Feature extractor with explicit extraction strategies and contracts."""

    def __init__(
        self,
        backbone: torch.nn.Module,
        layers_to_extract_from: list[str],
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        if not layers_to_extract_from:
            raise ValueError("layers_to_extract_from must not be empty.")
        self.backbone = backbone
        self.layers_to_extract_from = list(layers_to_extract_from)
        self.dtype = dtype
        self._strategy = self._build_strategy()

    def _build_strategy(self) -> _ExtractorStrategy:
        child_names = {name for name, _ in self.backbone.named_children()}
        if all(layer in child_names for layer in self.layers_to_extract_from):
            return _SequentialChildrenStrategy(self.backbone, self.layers_to_extract_from)
        return _HookBasedStrategy(self.backbone, self.layers_to_extract_from)

    def forward(
        self,
        x: torch.Tensor,
        return_dict: bool = True,
    ) -> dict[str, torch.Tensor] | list[torch.Tensor]:
        x = x.to(self.dtype)
        outputs = self._strategy.extract(x)
        if return_dict:
            return outputs
        return [outputs[layer] for layer in self.layers_to_extract_from]

    def close(self) -> None:
        self._strategy.close()
