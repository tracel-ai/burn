#!/usr/bin/env python3

# used to generate model: onnx-tests/tests/where/where.onnx

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, condition, x, y):
        return torch.where(condition, x, y)


def main():
    # Set random seed for reproducibility
    torch.manual_seed(0)

    # Export to onnx
    model = Model()
    model.eval()
    device = torch.device("cpu")
    onnx_name = "where.onnx"
    x = torch.ones(2, 2, device=device)
    y = torch.zeros(2, 2, device=device)
    mask = torch.tensor([[True, False], [False, True]], device=device)
    test_input = (mask, x, y)

    torch.onnx.export(model, (test_input), onnx_name, verbose=False, opset_version=16)

    print(f"Finished exporting model to {onnx_name}")

    # Output some test data for use in the test
    print(f"Test input data: {test_input}")
    output = model.forward(*test_input)
    print(f"Test output data: {output}")


if __name__ == "__main__":
    main()