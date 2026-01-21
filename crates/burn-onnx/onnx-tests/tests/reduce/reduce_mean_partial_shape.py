#!/usr/bin/env python3

# Regression test for partial static_shape in ReduceMean
# used to generate model: onnx-tests/tests/reduce/reduce_mean_partial_shape.onnx

import torch
import torch.nn as nn
import onnx
from onnx import helper, TensorProto


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):
        # ReduceMean with keepdims=True on last dimension
        # This simulates ALBERT's LayerNorm pattern where only the last dim is known
        return torch.mean(x, dim=2, keepdim=True)


def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Export to onnx with dynamic shapes
    model = Model()
    model.eval()
    device = torch.device("cpu")
    onnx_name = "reduce_mean_partial_shape.onnx"

    # Create test input with shape [1, 4, 8]
    test_input = torch.randn(1, 4, 8, device=device)

    # Export with dynamic axes to create partial shape
    torch.onnx.export(
        model,
        test_input,
        onnx_name,
        verbose=False,
        opset_version=16,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch', 1: 'seq'},  # Only last dim is static
            'output': {0: 'batch', 1: 'seq'}
        }
    )

    print(f"Finished exporting model to {onnx_name}")

    # Output some test data for use in the test
    print(f"Test input shape: {test_input.shape}")
    output = model.forward(test_input)
    print(f"Test output shape: {output.shape}")
    print(f"Test input data:\n{test_input}")
    print(f"Test output data:\n{output}")


if __name__ == "__main__":
    main()
