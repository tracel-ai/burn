#!/usr/bin/env python3

# used to generate model: transpose.onnx

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):
        return x.permute(2, 0, 1)


def main():

    # Set seed for reproducibility
    torch.manual_seed(42)

    torch.set_printoptions(precision=8)

    # Export to onnx
    model = Model()
    model.eval()
    device = torch.device("cpu")

    file_name = "transpose.onnx"
    test_input = torch.arange(24, dtype=torch.float, device=device).reshape(2, 3, 4)
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
