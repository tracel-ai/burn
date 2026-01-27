#!/usr/bin/env python3

# used to generate model: onnx-tests/tests/add/add.onnx

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, k, m):

        a = k / m

        x = x / a

        return x


def main():

    # Export to onnx
    model = Model()
    model.eval()
    device = torch.device("cpu")
    onnx_name = "div.onnx"
    dummy_input = torch.randn(1, 2, 3, 4, device=device)

    scalar1, scalar2 = 9.0, 3.0

    torch.onnx.export(model, (dummy_input, scalar1, scalar2), onnx_name,
                      verbose=False, opset_version=16)

    print("Finished exporting model to {}".format(onnx_name))

    # Output some test data for use in the test
    test_input = torch.tensor([[[[3.0, 6.0, 6.0, 9.0]]]])

    print("Test input data: {}, {}, {}".format(test_input, scalar1, scalar2))
    output = model.forward(test_input, scalar1, scalar2)
    print("Test output data: {}".format(output))


if __name__ == '__main__':
    main()
