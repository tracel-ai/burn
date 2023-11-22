#!/usr/bin/env python3

# used to generate model: onnx-tests/tests/sqrt/sqrt.onnx

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):
        return torch.sqrt(x)


def main():
    # Set random seed for reproducibility
    torch.manual_seed(0)

    # Export to onnx
    model = Model()
    model.eval()
    device = torch.device("cpu")
    onnx_name = "sqrt.onnx"
    dummy_input = torch.randn(1, 4, 9, 25, device=device)

    torch.onnx.export(model, (dummy_input), onnx_name,
                      verbose=False, opset_version=16)

    print("Finished exporting model to {}".format(onnx_name))

    # Output some test data for use in the test
    test_input = torch.tensor([[[[1.0, 4.0, 9.0, 25.0]]]])

    print("Test input data: {}".format(test_input))
    output = model.forward(test_input)
    print("Test output data: {}".format(output))


if __name__ == '__main__':
    main()
