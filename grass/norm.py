from typing import Final
from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from grass.utils import scatter_reduce


def get_norm_module(
    norm_type: str, dim: int | None = None, eps: float = 1e-7
) -> nn.Module:
    if norm_type == "batch":
        assert dim is not None
        return BatchNorm(dim=dim, eps=eps)
    elif norm_type == "batch_rms":
        assert dim is not None
        return BatchNorm(dim=dim, subtract_mean=False, eps=eps)
    elif norm_type == "layer":
        return NonBatchNorm(mode="layer", eps=eps)
    elif norm_type == "layer_rms":
        return NonBatchNorm(mode="layer", subtract_mean=False, eps=eps)
    elif norm_type == "instance":
        return NonBatchNorm(mode="instance", eps=eps)
    elif norm_type == "instance_rms":
        return NonBatchNorm(mode="instance", subtract_mean=False, eps=eps)
    elif norm_type == "transformer_rms":
        return TransformerRMSNorm(eps=eps)
    else:
        raise ValueError(f"invalid norm_type: {norm_type}")


class Norm(nn.Module, ABC):
    @property
    @abstractmethod
    def is_shift_invariant(self) -> bool:
        raise NotImplementedError()

    def toggle_track_and_use_running_stats(self, enable: bool) -> None:
        raise RuntimeError(
            "this normalization module does not track or use running stats"
        )


class BatchNorm(Norm):
    def __init__(self, dim: int, subtract_mean: bool = True, eps: float = 1e-7) -> None:
        assert dim >= 1
        assert eps > 0.0

        super().__init__()
        self.subtract_mean: Final = subtract_mean
        self.eps: Final = eps

        self.track_and_use_running_stats = False
        self.register_buffer("num_batches_tracked", torch.zeros(1, dtype=torch.long))
        self.register_buffer("running_second_moment", torch.ones(dim))
        if subtract_mean:
            self.register_buffer("running_first_moment", torch.zeros(dim))

    def toggle_track_and_use_running_stats(self, enable: bool) -> None:
        if enable and not self.track_and_use_running_stats:
            self.reset_running_stats()

        self.track_and_use_running_stats = enable

    @torch.no_grad()
    def reset_running_stats(self) -> None:
        nn.init.zeros_(self.num_batches_tracked)
        nn.init.ones_(self.running_second_moment)
        if self.subtract_mean:
            nn.init.zeros_(self.running_first_moment)

    def forward(
        self,
        input: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        update_factor = 0.0
        if self.track_and_use_running_stats and self.training:
            self.num_batches_tracked.add_(1)
            update_factor = self.num_batches_tracked.reciprocal()

        input = self._debias(input=input, update_factor=update_factor)
        input = self._descale(input=input, update_factor=update_factor)

        return input

    def _debias(self, input: torch.Tensor, update_factor: torch.Tensor) -> torch.Tensor:
        if not self.subtract_mean:
            return input

        first_moment = input.mean(dim=0)
        if self.track_and_use_running_stats:
            if self.training:
                self.running_first_moment.mul_(1.0 - update_factor)
                self.running_first_moment.addcmul_(first_moment, update_factor)
            else:
                first_moment = self.running_first_moment

        return input - first_moment

    def _descale(
        self, input: torch.Tensor, update_factor: torch.Tensor
    ) -> torch.Tensor:
        second_moment = input.square().mean(dim=0)
        if self.subtract_mean:
            num_samples = input.size(0)
            assert num_samples >= 2
            bessel_correction_factor = num_samples / (num_samples - 1)
            second_moment = bessel_correction_factor * second_moment

        if self.track_and_use_running_stats:
            if self.training:
                self.running_second_moment.mul_(1.0 - update_factor)
                self.running_second_moment.addcmul_(second_moment, update_factor)
            else:
                second_moment = self.running_second_moment

        return input * torch.rsqrt(second_moment + self.eps)

    @property
    def is_shift_invariant(self) -> bool:
        return self.subtract_mean


class FixedDistributionBatchNorm(BatchNorm):
    def __init__(
        self, dim: int, subtract_mean: bool = True, eps: float = 1e-38
    ) -> None:
        super().__init__(dim=dim, subtract_mean=subtract_mean, eps=eps)
        super().toggle_track_and_use_running_stats(enable=True)

    def toggle_track_and_use_running_stats(self, enable: bool) -> None:
        raise RuntimeError(
            "this normalization module always tracks and uses running stats"
        )


class NonBatchNorm(Norm):
    def __init__(
        self, mode: str, subtract_mean: bool = True, eps: float = 1e-7
    ) -> None:
        assert mode in ["layer", "instance"]
        assert eps > 0.0

        super().__init__()
        self.mode: Final = mode
        self.subtract_mean: Final = subtract_mean
        self.eps: Final = eps

    def forward(
        self, input: torch.Tensor, batch: torch.Tensor, num_graphs: int, **kwargs
    ) -> torch.Tensor:
        if not self.subtract_mean:
            mean_input = self._get_mean(input=input, batch=batch, num_graphs=num_graphs)
            input = input - mean_input

        squared_input = input.square()
        mean_squared_input = self._get_mean(
            input=squared_input, batch=batch, num_graphs=num_graphs
        )
        input = input * torch.rsqrt(mean_squared_input + self.eps)

        return input

    def _get_mean(
        self, input: torch.Tensor, batch: torch.Tensor, num_graphs: int
    ) -> torch.Tensor:
        if self.mode == "layer":
            input = input.mean(dim=-1, keepdim=True)

        return scatter_reduce(
            input=input,
            index=batch,
            num_bins=num_graphs,
            reduction="mean",
            broadcast_index=True,
            gather=True,
        )

    @property
    def is_shift_invariant(self) -> bool:
        return self.subtract_mean


class TransformerRMSNorm(Norm):
    def __init__(self, eps: float = 1e-7) -> None:
        assert eps > 0.0

        super().__init__()
        self.eps: Final = eps

    def forward(self, input: torch.Tensor, **kwargs) -> torch.Tensor:
        return input * torch.rsqrt(input.square().mean(dim=-1, keepdim=True) + self.eps)

    @property
    def is_shift_invariant(self) -> bool:
        return False
