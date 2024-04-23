#!/usr/bin/env python3

# used to generate model: onnx-tests/tests/matmul/matmul.onnx

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, a, b, c, d):
        return torch.matmul(a, b), torch.matmul(c, d), torch.matmul(d, c)


def main():
    # Set random seed for reproducibility
    torch.manual_seed(0)

    # Export to onnx
    model = Model()
    model.eval()
    device = torch.device("cpu")
    onnx_name = "matmul.onnx"
    a = torch.arange(24, dtype=torch.float, device=device).reshape(1, 2, 3, 4)
    b = torch.arange(16, dtype=torch.float, device=device).reshape(1, 2, 4, 2)
    c = torch.arange(96, dtype=torch.float, device=device).reshape(2, 3, 4, 4)
    d = torch.arange(4, dtype=torch.float, device=device)
    test_input = (a, b, c, d)

    torch.onnx.export(model, test_input, onnx_name, verbose=False, opset_version=16)

    print(f"Finished exporting model to {onnx_name}")

    # Output some test data for use in the test
    print(f"Test input data: {test_input}")
    output = model.forward(*test_input)
    print(f"Test output data: {output}")


if __name__ == "__main__":
    main()
