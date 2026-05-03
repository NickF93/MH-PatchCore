"""Plugin-local streaming covariance estimator for transform fitting."""

from __future__ import annotations

import torch


class StreamingCovariance:
    """Streaming covariance accumulator using stable merge updates."""

    def __init__(
        self,
        *,
        num_features: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        if num_features <= 0:
            raise ValueError("num_features must be a positive integer.")
        self.num_features = int(num_features)
        self._dtype = dtype
        self._device = device
        self.count = torch.tensor(0.0, dtype=dtype, device=device)
        self.mean = torch.zeros(self.num_features, dtype=dtype, device=device)
        self.m2 = torch.zeros(
            (self.num_features, self.num_features),
            dtype=dtype,
            device=device,
        )

    def _prepare_batch(self, batch: torch.Tensor) -> torch.Tensor:
        if batch.ndim < 2:
            raise ValueError("batch must have shape [batch, features...]")
        prepared = batch.to(device=self._device, dtype=self._dtype)
        if prepared.ndim > 2:
            prepared = prepared.reshape(prepared.shape[0], -1)
        if int(prepared.shape[1]) != self.num_features:
            raise ValueError(
                f"Expected {self.num_features} features, got {int(prepared.shape[1])}."
            )
        return prepared

    @torch.no_grad()
    def update(self, batch: torch.Tensor) -> None:
        prepared = self._prepare_batch(batch)
        batch_size = int(prepared.shape[0])
        if batch_size == 0:
            return
        batch_size_f = torch.tensor(
            float(batch_size),
            dtype=self._dtype,
            device=self._device,
        )
        batch_mean = prepared.mean(dim=0)
        centered = prepared - batch_mean
        batch_m2 = centered.t().matmul(centered)
        delta = batch_mean - self.mean
        new_count = self.count + batch_size_f
        mean = self.mean + delta * (batch_size_f / new_count)
        m2 = self.m2 + batch_m2 + torch.outer(delta, delta) * (
            self.count * batch_size_f / new_count
        )
        self.mean.copy_(mean)
        self.m2.copy_(m2)
        self.count.copy_(new_count)

    def covariance(self, *, unbiased: bool = True) -> torch.Tensor:
        if float(self.count.item()) < 2.0:
            return torch.zeros(
                (self.num_features, self.num_features),
                dtype=self._dtype,
                device=self._device,
            )
        denom = self.count - 1.0 if unbiased else self.count
        return self.m2 / denom

