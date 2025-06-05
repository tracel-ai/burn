#!/usr/bin/env python3

# Used to generate model: onnx-tests/tests/round/round.onnx

import torch
import torch.nn as nn
import onnx

class RoundModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.round(x)

def main():
    model = RoundModel()
    model.eval()

    test_input = torch.tensor([-0.5, 1.5, 2.1])

    onnx_file = "round.onnx"

    torch.onnx.export(
        model,
        test_input,
        onnx_file,
        opset_version=16,
    )

    print(f"Finished exporting model to {onnx_file}")
    print(f"Test input data of ones: {test_input}")
    print(f"Test input data shape of ones: {test_input.shape}")
    output = model.forward(test_input)
    print(f"Test output data shape: {output.shape}")
    print(f"Test output: {output}")


if __name__ == '__main__':
    main()
