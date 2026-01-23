#!/usr/bin/env python3

# used to generate model: onnx-tests/tests/argmin_1d/argmin_1d.onnx

import torch
import torch.nn as nn
import numpy as np
import onnx
from onnx import TensorProto, helper

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):
        # keepdims=False on 1D tensor should output scalar
        y = torch.argmin(input=x, dim=0, keepdim=False)
        return y

def main():
    # Export to onnx
    model = Model()
    model.eval()
    device = torch.device("cpu")
    onnx_name = "argmin_1d.onnx"
    
    # Create dummy 1D input
    dummy_input = torch.randn(5, device=device)
    
    torch.onnx.export(model, dummy_input, onnx_name,
                      verbose=False, opset_version=16,
                      input_names=['input'],
                      output_names=['output'],
                      dynamic_axes={'input': {0: 'dim0'}})
    
    print("Finished exporting model to {}".format(onnx_name))
    
    # Output some test data for use in the test
    test_input = torch.tensor([1.0, 3.0, 2.0, 5.0, 4.0], device=device)
    print("Test input data: {}".format(test_input))
    output = model.forward(test_input)
    print("Test output (argmin index): {}".format(output.item()))
    print("Test output shape: {} (scalar)".format(output.shape))

if __name__ == '__main__':
    main()