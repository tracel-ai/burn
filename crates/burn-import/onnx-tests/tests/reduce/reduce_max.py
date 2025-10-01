#!/usr/bin/env python3

# used to generate model: onnx-tests/tests/reduce/reduce_max.onnx

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):
        return (
            # ReduceMax, keepdims=0, axes=None
            torch.max(x),
            # ReduceMax, keepdims=0, axes=[0, 1]
            torch.amax(x, dim=None, keepdim=False),
            # ReduceMax, keepdims=1, axes=[1]
            torch.max(x, dim=1, keepdim=True).values,
            # ReduceMax, keepdims=1, axes=[-1]
            torch.max(x, dim=-1, keepdim=True).values,
            # ReduceMax, keepdims=0, axes=[0]
            torch.max(x, dim=0, keepdim=False).values,
            # ReduceMax, keepdims=0, axes=[0, 2]
            torch.amax(x, dim=(0, 2), keepdim=False),
        )


def main():
    # Set random seed for reproducibility
    torch.manual_seed(0)

    # Export to onnx
    model = Model()
    model.eval()
    device = torch.device("cpu")
    onnx_name = "reduce_max.onnx"
    test_input = torch.tensor([[[
        [1.0, 4.0, 9.0, 25.0],
        [2.0, 5.0, 10.0, 26.0],
    ]]], device=device)

    torch.onnx.export(model, test_input, onnx_name, 
                      verbose=False, opset_version=16)

    print(f"Finished exporting model to {onnx_name}")

    # Output some test data for use in the test
    print(f"Test input data:\n{test_input}")
    output = model.forward(test_input)
    print("Test output data:", *output, sep = "\n")


if __name__ == "__main__":
    main()
