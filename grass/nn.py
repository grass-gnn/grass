from typing import Final

import torch
from torch import nn
import torch.nn.functional as F


class KaimingLinear(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, bias: bool = True) -> None:
        super().__init__()

        self.linear = nn.Linear(
            in_features=input_dim, out_features=output_dim, bias=bias
        )

        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self, gain: float = 1.0) -> None:
        nn.init.kaiming_uniform_(self.linear.weight, nonlinearity="linear")
        self.linear.weight.mul_(gain)

        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.linear(input)


class GatedLinearUnit(nn.Module):
    def __init__(
        self, input_dim: int, output_dim: int, hidden_dim: int, bias: bool = True
    ) -> None:
        assert input_dim >= 1
        assert output_dim >= 1
        assert hidden_dim >= 1

        super().__init__()

        input_linear_output_dim = 2 * hidden_dim
        self.input_linear: Final = KaimingLinear(
            input_dim=input_dim,
            output_dim=input_linear_output_dim,
            bias=True,
        )

        output_linear_input_dim = input_dim + hidden_dim
        self.output_linear: Final = KaimingLinear(
            input_dim=output_linear_input_dim, output_dim=output_dim, bias=bias
        )

        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        self.input_linear.reset_parameters()
        self.output_linear.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        (input, gate) = self.input_linear(x).tensor_split(sections=2, dim=-1)
        hidden_activation = input * F.mish(gate)
        return self.output_linear(torch.cat((x, hidden_activation), dim=-1))
