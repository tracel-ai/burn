#!/usr/bin/env python3

# used to generate model: sub_int.onnx

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        # TODO enable this after https://github.com/tracel-ai/burn/issues/665 is fixed
        # Declare a constant int tensor with ones
        # self.a = torch.ones(1, 1, 1, 4)

        # Declare a scalar
        self.b = 9
        super(Model, self).__init__()

    def forward(self, x, k):

        # Subtract a tensor from a tensor input
        x = x - x

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
    onnx_name = "sub_int.onnx"

    test_input = torch.tensor([[[[1, 2, 3, 4]]]], device=device)
    scalar = 3

    torch.onnx.export(
        model, (test_input, scalar), onnx_name, verbose=False, opset_version=16
    )

    print("Finished exporting model to {}".format(onnx_name))

    print("Test input data: {}, {}".format(test_input, scalar))
    output = model.forward(test_input, scalar)
    print("Test output data: {}".format(output))


if __name__ == "__main__":
    main()
