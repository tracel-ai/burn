#!/usr/bin/env python3

# used to generate model: onnx-tests/tests/argmin/argmin.onnx

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, argmin_dim: int = 0):
        super(Model, self).__init__()
        self._argmin_dim = argmin_dim

    def forward(self, x):
        # Note: only keepdim=True is supported in burn
        y = torch.argmin(input=x, dim=self._argmin_dim, keepdim=True)
        return y


def main():

    # Export to onnx
    model = Model(1)
    model.eval()
    device = torch.device("cpu")
    onnx_name = "argmin.onnx"
    dummy_input = torch.randn((3, 4), device=device)
    torch.onnx.export(model, dummy_input, onnx_name,
                      verbose=False, opset_version=16)

    print("Finished exporting model to {}".format(onnx_name))

    # Output some test data for use in the test
    test_input = torch.randn((2, 3), device=device)
    print("Test input data shape: {}".format(test_input.shape))
    print("Test input data:\n{}".format(test_input))
    output = model.forward(test_input)

    print("Test output data shape: {}".format(output.shape))
    print("Test output data:\n{}".format(output))


if __name__ == '__main__':
    main()
