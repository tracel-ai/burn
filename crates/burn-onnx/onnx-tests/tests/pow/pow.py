#!/usr/bin/env python3

# used to generate model: onnx-tests/tests/add/add.onnx

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # self.b = 5.0

    def forward(self, x, k):
        # raise a tensor to tensor power
        x = x.pow(x)

        # raise a scalar constant to a power of a scalar
        # d = torch.pow(self.b, k)

        # raise a tensor input to a power of a scalar
        x = torch.pow(x, k)

        return x


def main():
    # Export to onnx
    model = Model()
    model.eval()
    device = torch.device("cpu")
    onnx_name = "pow.onnx"
    dummy_input = torch.randn(1, 2, 3, 4, dtype=torch.float32, device=device)

    scalar = 2.0

    torch.onnx.export(
        model, (dummy_input, scalar), onnx_name, verbose=False, opset_version=16
    )

    print(f"Finished exporting model to {onnx_name}")

    # Output some test data for use in the test
    test_input = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]])

    print(f"Test input data: {test_input}, {scalar}")
    output = model.forward(test_input, scalar)
    print(f"Test output data: {output}")


if __name__ == "__main__":
    main()
