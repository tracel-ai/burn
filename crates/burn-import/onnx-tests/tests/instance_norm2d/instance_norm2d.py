#!/usr/bin/env python3

# used to generate model: onnx-tests/tests/instance_norm2d/instance_norm2d.onnx

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.norm2 = nn.InstanceNorm2d(3)

    def forward(self, x):
        return self.norm2(x)


def main():
    # Set random seed for reproducibility
    torch.manual_seed(0)

    # Export to onnx
    model = Model()
    model.eval()
    device = torch.device("cpu")

    onnx_name = "instance_norm2d.onnx"
    test_input = torch.arange(48, dtype=torch.float, device=device).reshape(2, 2, 3, 4)
    
    torch.onnx.export(model, test_input, onnx_name, verbose=False, opset_version=16)

    print(f"Finished exporting model to {onnx_name}")

    # Output some test data for use in the test
    print(f"Test input data: {test_input}")
    output = model.forward(test_input)
    print(f"Test output data: {output}")


if __name__ == "__main__":
    main()
