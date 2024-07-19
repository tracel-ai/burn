#!/usr/bin/env python3

# used to generate model: squeeze.onnx

import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.dims = 2

    def forward(self, x):
        # With no second argument, squeeze removes all singleton dimensions
        x = torch.squeeze(x, self.dims)
        return x


def main():
    # Set seed for reproducibility
    torch.manual_seed(42)

    torch.set_printoptions(precision=8)

    # Export to onnx
    model = Model()
    model.eval()
    device = torch.device("cpu")

    test_input = torch.randn(3, 4, 1, 5, 1, device=device)

    # Export to ONNX
    torch.onnx.export(model, test_input, "squeeze_opset16.onnx", verbose=False, opset_version=16)
    torch.onnx.export(model, test_input, "squeeze_opset13.onnx", verbose=False, opset_version=13)

    print(f"Finished exporting model to 16 and 13")

    # Output some test data for use in the test
    output = model(test_input)
    print(f"Test input data: {test_input}")
    print(f"Test input data shape: {test_input.shape}")
    print(f"Test output data shape: {output.shape}")
    print(f"Test output: {output}")


if __name__ == "__main__":
    main()
