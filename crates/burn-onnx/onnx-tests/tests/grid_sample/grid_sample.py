#!/usr/bin/env python3
"""
Generate ONNX model for GridSample operation testing.

GridSample performs spatial sampling using normalized grid coordinates.
This test uses bilinear interpolation mode.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GridSampleModel(nn.Module):
    """A simple model that uses grid_sample with bilinear interpolation."""

    def __init__(self):
        super().__init__()

    def forward(self, x, grid):
        # align_corners=False, mode='bilinear', padding_mode='zeros' (defaults)
        return F.grid_sample(x, grid, mode='bilinear', padding_mode='zeros', align_corners=False)


def main():
    torch.manual_seed(42)

    # Create model
    model = GridSampleModel()
    model.eval()

    # Create test inputs
    # Input: (N, C, H_in, W_in) = (1, 1, 4, 4)
    x = torch.randn(1, 1, 4, 4)
    # Grid: (N, H_out, W_out, 2) = (1, 3, 3, 2) with normalized coordinates [-1, 1]
    grid = torch.rand(1, 3, 3, 2) * 2 - 1  # Random coordinates in [-1, 1]

    # Export to ONNX
    torch.onnx.export(
        model,
        (x, grid),
        "grid_sample.onnx",
        input_names=["input", "grid"],
        output_names=["output"],
        opset_version=16,
        dynamic_axes=None,
    )

    # Run inference for verification
    output = model(x, grid)

    print(f"Input shape: {x.shape}")
    print(f"Grid shape: {grid.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Input:\n{x}")
    print(f"Grid:\n{grid}")
    print(f"Output:\n{output}")


if __name__ == "__main__":
    main()
