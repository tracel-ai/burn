#!/usr/bin/env python3

# used to generate model: onnx-tests/tests/max/max.onnx

import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):
        return torch.max(x)

def main():
    # Set seed for reproducibility
    torch.manual_seed(42)
    torch.set_printoptions(precision=8)

    # Export to onnx
    model = Model()
    model.eval()
    device = torch.device("cpu")

    onnx_name = "max.onnx"

    test_input = torch.randn(4, 4, device=device)
    torch.onnx.export(model, test_input, onnx_name, verbose=False, opset_version=16)

    print("Finished exporting model to {}".format(onnx_name))

    print("Test input data: {}".format(test_input))
    output = model.forward(test_input)
    print("Test output data: {}".format(output))

if __name__ == '__main__':
    main()
