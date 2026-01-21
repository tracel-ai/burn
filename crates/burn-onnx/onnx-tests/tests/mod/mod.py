#!/usr/bin/env python3

# used to generate model: onnx-tests/tests/mod/mod.onnx

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y):
        # Compute the modulo operation
        z = torch.fmod(x, y)
        return z


def main():
    # Export to onnx
    model = Model()
    model.eval()
    device = torch.device("cpu")
    onnx_name = "mod.onnx"

    # Create dummy inputs with proper shapes
    dummy_x = torch.randn(2, 3, 4, device=device)
    dummy_y = torch.randn(2, 3, 4, device=device)

    torch.onnx.export(model, (dummy_x, dummy_y), onnx_name,
                      verbose=False, opset_version=16)

    print("Finished exporting model to {}".format(onnx_name))

    # Output some test data for use in the test
    test_x = torch.tensor([[[[5.3, -5.3, 7.5, -7.5]]]])
    test_y = torch.tensor([[[[2.0, 2.0, 3.0, 3.0]]]])

    print("Test input x: {}".format(test_x))
    print("Test input y: {}".format(test_y))
    output = model.forward(test_x, test_y)
    print("Test output: {}".format(output))


if __name__ == '__main__':
    main()