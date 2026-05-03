"""Contract for runtime plugin bundle wiring."""

from __future__ import annotations

from typing import Protocol

from .plugins.dataloader.contracts import DataLoaderPlugin
from .model_plugin_bundle_contract import ModelPluginBundle


class RuntimePluginBundle(Protocol):
    """Contract-only protocol for runtime plugin composition."""

    @property
    def dataloader_plugin(self) -> DataLoaderPlugin: ...

    @property
    def model_plugin_bundle(self) -> ModelPluginBundle: ...
