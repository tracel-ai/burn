#!/usr/bin/env python3

# used to generate model: onnx-tests/tests/add/add.onnx

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        # Declare a constant float tensor with ones
        self.a = torch.ones(1, 1, 1, 4)

        # Declare a scalar
        self.b = 9.0
        super(Model, self).__init__()

    def forward(self, x, k):

        # Subtract a constant tensor from a tensor input
        x = x - self.a

        # Subtract a scalar constant from a scalar input
        d = k - self.b

        # Subtract a scalar from a tensor
        x = x - d

        # Subtract a tensor from a scalar
        x = d - x

        return x


def main():

    # Export to onnx
    model = Model()
    model.eval()
    device = torch.device("cpu")
    onnx_name = "sub.onnx"
    dummy_input = torch.randn(1, 2, 3, 4, device=device)

    scalar = 3.0

    torch.onnx.export(
        model, (dummy_input, scalar), onnx_name, verbose=False, opset_version=16
    )

    print("Finished exporting model to {}".format(onnx_name))

    # Output some test data for use in the test
    test_input = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]])

    print("Test input data: {}, {}".format(test_input, scalar))
    output = model.forward(test_input, scalar)
    print("Test output data: {}".format(output))


if __name__ == "__main__":
    main()
