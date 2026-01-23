#!/usr/bin/env python3

# used to generate model: onnx-tests/tests/add/add.onnx

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):

        # TODO enable this after https://github.com/tracel-ai/burn/issues/665 is fixed
        # Declare a constant int tensor with ones
        # self.a = torch.ones(1, 1, 1, 4, dtype=torch.int32)

        # Declare a scalar
        self.b = 5
        super(Model, self).__init__()

    def forward(self, x, k):

        # Add tensor inputs
        x = x + x

        # Add a scalar constant and a scalar input
        d = self.b + k

        # Add a tensor and a scalar
        x = x + d

        return x


def main():

    # set seed for reproducibility
    torch.manual_seed(0)

    # Export to onnx
    model = Model()
    model.eval()
    device = torch.device("cpu")
    onnx_name = "add_int.onnx"
    # Output some test data for use in the test
    test_input = torch.tensor([[[[1, 2, 3, 4]]]], dtype=torch.int32)

    scalar = 2

    torch.onnx.export(model, (test_input, scalar), onnx_name,
                      verbose=False, opset_version=16)

    print("Finished exporting model to {}".format(onnx_name))

    print("Test input data: {}, {}".format(test_input, scalar))
    output = model.forward(test_input, scalar)
    print("Test output data: {}".format(output))


if __name__ == '__main__':
    main()
