#!/usr/bin/env python3

import torch
from torch import nn, Tensor
from safetensors.torch import save_file


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 6, 3, bias=False)
        self.bn = nn.BatchNorm2d(6)
        self.layer = nn.Sequential(ConvBlock(6, 6), ConvBlock(6, 6))

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.layer(x)

        return x


def main():
    torch.set_printoptions(precision=8)
    torch.manual_seed(42)

    model = Model()

    input = torch.rand(1, 3, 4, 4)
    model(input)  # condition batch norm
    model.eval()

    with torch.no_grad():
        print(f"Input shape: {input.shape}")
        print("Input type: {}", input.dtype)
        print(f"Input: {input}")
        output = model(input)

    print(f"Output: {output}")
    print(f"Output Shape: {output.shape}")

    save_file(model.state_dict(), "key_remap_chained.safetensors")


if __name__ == "__main__":
    main()
