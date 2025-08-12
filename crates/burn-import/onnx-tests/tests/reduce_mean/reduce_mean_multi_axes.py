#!/usr/bin/env python3

# used to generate model: onnx-tests/tests/reduce_mean/reduce_mean_multi_axes.onnx

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):
        # ReduceMean, keepdims=0, axes=[0, 2]
        # Shape: [2, 3, 4] -> reduce on dims 0 and 2 -> [3]
        return torch.mean(x, dim=(0, 2), keepdim=False)


def main():
    # Set random seed for reproducibility
    torch.manual_seed(0)

    # Export to onnx
    model = Model()
    model.eval()
    device = torch.device("cpu")
    onnx_name = "reduce_mean_multi_axes.onnx"
    # Shape: [2, 3, 4] -> after reduction on dims [0, 2] -> [3]
    test_input = torch.randn(2, 3, 4, device=device)

    torch.onnx.export(model, test_input, onnx_name, verbose=False, opset_version=16)

    print(f"Finished exporting model to {onnx_name}")

    # Output some test data for use in the test
    print(f"Test input shape: {test_input.shape}")
    print(f"Test input data: {test_input}")
    output = model.forward(test_input)
    print(f"Test output shape: {output.shape}")
    print(f"Test output data: {output}")


if __name__ == "__main__":
    main()