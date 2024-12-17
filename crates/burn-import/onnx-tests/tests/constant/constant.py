#!/usr/bin/env python3

import torch
import torch.nn as nn

class ConstantModel(nn.Module):
    def __init__(self):
        super(ConstantModel, self).__init__()

    def forward(self, x):
        # '2.0' should result in a constant node
        return x + 2.0

def main():
    model = ConstantModel()
    model.eval()
    device = torch.device("cpu")
    onnx_name = "constant.onnx"

    # Dummy input for export
    dummy_input = torch.randn(3, 4, device=device)
    torch.onnx.export(model, dummy_input, onnx_name,
                      verbose=False, opset_version=16, do_constant_folding=False)

    print("Finished exporting model to {}".format(onnx_name))

    # Output some test data for use in testing
    input = torch.randn(2, 3, device=device)
    print("Test input:", input)
    print("Test input data shape: {}".format(input.shape))
    output = model.forward(input)
    print("Test output:", output)
    print("Test output data shape: {}".format(output.shape))

if __name__ == '__main__':
    main()
