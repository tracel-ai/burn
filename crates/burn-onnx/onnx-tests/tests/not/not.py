#!/usr/bin/env python3

# used to generate model: onnx-tests/tests/not/not.onnx

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):
        return torch.logical_not(x)


def main():
    # Set random seed for reproducibility
    torch.manual_seed(0)

    # Export to onnx
    model = Model()
    model.eval()
    device = torch.device("cpu")
    onnx_name = "not.onnx"
    test_input = torch.tensor([[[[True, False, True, False]]]], device=device)

    # NOTE: torch exports logical_not with a cast node even if the input is already bool
    # https://github.com/pytorch/pytorch/blob/main/torch/onnx/symbolic_opset9.py#L2204-L2207
    torch.onnx.export(model, test_input, onnx_name, verbose=False, opset_version=16)

    print(f"Finished exporting model to {onnx_name}")

    # Output some test data for use in the test
    print(f"Test input data: {test_input}")
    output = model.forward(test_input)
    print(f"Test output data: {output}")


if __name__ == "__main__":
    main()
