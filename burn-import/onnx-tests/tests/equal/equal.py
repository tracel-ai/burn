#!/usr/bin/env python3

# used to generate model: equal.onnx

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        # Declare a constant float tensor with ones
        self.a = torch.ones(1, 1, 1, 4)

        # Declare a scalar
        self.b = 5.0
        super(Model, self).__init__()

    def forward(self, x, k):

        x = x == self.a

        k = k == self.b

        return x, k


def main():

    # Set seed for reproducibility
    torch.manual_seed(42)

    # Export to onnx
    model = Model()
    model.eval()
    device = torch.device("cpu")
    onnx_name = "equal.onnx"
    input = torch.ones(1, 1, 1, 4, device=device)

    scalar = 2.0

    torch.onnx.export(model, (input, scalar), onnx_name,
                      verbose=False, opset_version=16)

    print("Finished exporting model to {}".format(onnx_name))

    print("Test input data: {}, {}".format(input, scalar))
    output = model.forward(input, scalar)
    print("Test output data: {}".format(output))


if __name__ == '__main__':
    main()
