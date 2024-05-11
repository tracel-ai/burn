#!/usr/bin/env python3

# used to generate model: squeeze.onnx

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.axis = 2

    def forward(self, x):
        x = torch.squeeze(x, self.axis)
        return x


def main():
    # Set seed for reproducibility
    torch.manual_seed(42)

    torch.set_printoptions(precision=8)

    # Export to onnx
    model = Model()
    model.eval()
    device = torch.device("cpu")

    test_input = torch.randn(3, 4, 1, 5, device=device)
    model = Model()

    # Export to ONNX
    file_name = "squeeze.onnx"
    torch.onnx.export(model, test_input, file_name, verbose=False, opset_version=16)

    print(f"Finished exporting model to {file_name}")

    # Output some test data for use in the test
    output = model(test_input)
    print(f"Test input data: {test_input}")
    print(f"Test input data shape: {test_input.shape}")
    print(f"Test output data shape: {output.shape}")
    print(f"Test output: {output}")


if __name__ == "__main__":
    main()
