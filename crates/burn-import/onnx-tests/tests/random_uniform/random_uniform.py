#!/usr/bin/env python3

# used to generate model: random_uniform.onnx

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, _in):
        return torch.rand(2, 3)


def main():
    # Set seed for reproducibility
    torch.manual_seed(42)

    torch.set_printoptions(precision=8)

    # Export to onnx
    model = Model()
    model.eval()
    device = torch.device("cpu")

    file_name = "random_uniform.onnx"
    test_input = torch.empty(0)
    torch.onnx.export(model, test_input, file_name,
                      verbose=False, opset_version=16)

    print(f"Finished exporting model to {file_name}")

    # Output some test data for use in the test
    print(f"Test input data: {test_input}")
    print(f"Test input data shape: {test_input.shape}")
    output = model.forward(test_input)
    print(f"Test output data shape: {output.shape}")
    print(f"Test output data: {output}")


if __name__ == '__main__':
    main()