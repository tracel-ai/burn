#!/usr/bin/env python3

# used to generate model: onnx-tests/tests/add/add.onnx

import torch
import torch.nn as nn
from torch.onnx import OperatorExportTypes


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # self.b = 5.0

    def forward(self, x, k):
        # raise a tensor to tensor power
        x = x.pow(x)

        # raise a scalar constant to a power of a scalar
        # d = torch.pow(self.b, k).int()

        # raise a tensor input to a power of a scalar
        x = torch.pow(x, k)

        return x


def main():
    # Export to onnx
    model = Model()
    model.eval()
    device = torch.device("cpu")
    onnx_name = "pow_int.onnx"
    test_input = torch.tensor([[[[1, 2, 3, 4]]]], dtype=torch.int32)

    scalar = 2

    torch.onnx.export(
        model, (test_input, scalar), onnx_name, verbose=False, opset_version=16
    )

    print(f"Finished exporting model to {onnx_name}")

    print(f"Test input data: {test_input}, {scalar}")
    output = model.forward(test_input, scalar)
    print(f"Test output data: {output}")


if __name__ == "__main__":
    main()
