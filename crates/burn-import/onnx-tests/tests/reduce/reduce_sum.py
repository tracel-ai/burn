#!/usr/bin/env python3

# used to generate model: onnx-tests/tests/reduce/reduce_sum.onnx

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):
        return (
            # ReduceSum, keepdims=0, axes=None
            torch.sum(x),
            # ReduceSum, keepdims=0, axes=None
            torch.sum(x, dim=None, keepdim=False),
            # ReduceSum, keepdims=1, axes=[1]
            torch.sum(x, dim=1, keepdim=True),
            # ReduceSum, keepdims=1, axes=[-1]
            torch.sum(x, dim=-1, keepdim=True),
            # ReduceSum, keepdims=0, axes=[0]
            torch.sum(x, dim=0, keepdim=False),
            # ReduceSum, keepdims=0, axes=[0, 2]
            torch.sum(x, dim=(0, 2), keepdim=False),
        )


def main():
    # Set random seed for reproducibility
    torch.manual_seed(0)

    # Export to onnx
    model = Model()
    model.eval()
    device = torch.device("cpu")
    test_input = torch.tensor([[[
        [1.0, 4.0, 9.0, 25.0],
        [2.0, 5.0, 10.0, 26.0],
    ]]], device=device)

    torch.onnx.export(model, test_input, "reduce_sum.onnx", verbose=False, opset_version=16)

    print("Finished exporting model")

    # Output some test data for use in the test
    print(f"Test input data:\n{test_input}")
    output = model.forward(test_input)
    print("Test output data:", *output, sep = "\n")


if __name__ == "__main__":
    main()
