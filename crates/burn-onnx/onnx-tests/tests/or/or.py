#!/usr/bin/env python3

# used to generate model: onnx-tests/tests/or/or.onnx

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y):
        return torch.logical_or(x, y)


def main():
    # Set random seed for reproducibility
    torch.manual_seed(0)

    # Export to onnx
    model = Model()
    model.eval()
    device = torch.device("cpu")
    onnx_name = "or.onnx"
    test_input_x = torch.tensor([[[[False, False, True, True]]]], device=device)
    test_input_y = torch.tensor([[[[False, True, False, True]]]], device=device)

    # NOTE: torch exports logical_or with a cast node even if the input is already bool
    # https://github.com/pytorch/pytorch/blob/main/torch/onnx/symbolic_opset9.py#L2204-L2207
    torch.onnx.export(model, (test_input_x, test_input_y), onnx_name, verbose=False, opset_version=16)

    print(f"Finished exporting model to {onnx_name}")

    # Output some test data for use in the test
    print(f"Test input data: {test_input_x}, {test_input_y}")
    output = model.forward(test_input_x, test_input_y)
    print(f"Test output data: {output}")


if __name__ == "__main__":
    main()
