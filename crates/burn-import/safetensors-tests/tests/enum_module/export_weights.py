#!/usr/bin/env python3
import torch
from torch import nn, Tensor
from safetensors.torch import save_file


class DwsConv(nn.Module):
    """Depthwise separable convolution."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int) -> None:
        super().__init__()
        # Depthwise conv
        self.dconv = nn.Conv2d(
            in_channels, in_channels, kernel_size, groups=in_channels
        )
        # Pointwise conv
        self.pconv = nn.Conv2d(in_channels, out_channels, kernel_size=1, groups=1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.dconv(x)
        return self.pconv(x)


class Model(nn.Module):
    def __init__(self, depthwise: bool = False) -> None:
        super().__init__()
        self.conv = DwsConv(2, 2, 3) if depthwise else nn.Conv2d(2, 2, 3)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


def main():

    torch.set_printoptions(precision=8)
    torch.manual_seed(1)

    model = Model().to(torch.device("cpu"))

    save_file(model.state_dict(), "enum_depthwise_false.safetensors")

    input = torch.rand(1, 2, 5, 5)

    print("Depthwise is False")
    print("Input shape: {}", input.shape)
    print("Input: {}", input)
    output = model(input)
    print("Output: {}", output)
    print("Output Shape: {}", output.shape)

    print("Depthwise is True")
    model = Model(depthwise=True).to(torch.device("cpu"))
    save_file(model.state_dict(), "enum_depthwise_true.safetensors")

    print("Input shape: {}", input.shape)
    print("Input: {}", input)
    output = model(input)
    print("Output: {}", output)
    print("Output Shape: {}", output.shape)


if __name__ == "__main__":
    main()
