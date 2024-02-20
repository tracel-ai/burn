#!/usr/bin/env python3

# used to generate model: onnx-tests/tests/add/add.onnx

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        # Declare a constant float tensor
        self.a = torch.full((1, 1, 1, 4), 3.0)

        # Declare a scalar
        self.b = 7.0
        super(Model, self).__init__()

    def forward(self, x, k):

        # Multiply the input by the constant tensor
        x = x * self.a

        # Multiply the input scalar by the constant scalar
        d = k * self.b

        # Multiply the result of the previous multiplications
        x = x * d

        return x


def main():

    # Export to onnx
    model = Model()
    model.eval()
    device = torch.device("cpu")
    onnx_name = "mul.onnx"
    dummy_input = torch.randn(1, 2, 3, 4, device=device)

    scalar = 6.0

    torch.onnx.export(model, (dummy_input, scalar), onnx_name,
                      verbose=False, opset_version=16)

    print("Finished exporting model to {}".format(onnx_name))

    # Output some test data for use in the test
    test_input = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]])

    print("Test input data: {}, {}".format(test_input, scalar))
    output = model.forward(test_input, scalar)
    print("Test output data: {}".format(output))


if __name__ == '__main__':
    main()
