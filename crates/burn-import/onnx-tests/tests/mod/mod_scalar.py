#!/usr/bin/env python3

# used to generate model: onnx-tests/tests/mod/mod_scalar.onnx

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, scalar):
        # Compute the modulo operation with a scalar
        z = torch.fmod(x, scalar)
        return z


def main():
    # Export to onnx
    model = Model()
    model.eval()
    device = torch.device("cpu")
    onnx_name = "mod_scalar.onnx"

    # Create dummy inputs
    dummy_x = torch.randn(1, 2, 3, 4, device=device)
    dummy_scalar = 2.0

    torch.onnx.export(model, (dummy_x, dummy_scalar), onnx_name,
                      verbose=False, opset_version=16)

    print("Finished exporting model to {}".format(onnx_name))

    # Output some test data for use in the test
    test_x = torch.tensor([[[[5.3, -5.3, 7.5, -7.5]]]])
    test_scalar = 2.0

    print("Test input x: {}".format(test_x))
    print("Test scalar: {}".format(test_scalar))
    output = model.forward(test_x, test_scalar)
    print("Test output: {}".format(output))


if __name__ == '__main__':
    main()