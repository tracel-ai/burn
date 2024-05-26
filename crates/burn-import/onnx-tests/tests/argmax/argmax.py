#!/usr/bin/env python3

# used to generate model: onnx-tests/tests/concat/concat.onnx

import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, argmax_dim: int = 0):
        super(Model, self).__init__()
        self._argmax_dim = argmax_dim

    def forward(self, x):
        # Concatenate along the channel dimension
        y = torch.argmax(input=x, dim=self._argmax_dim)
        return y.to(float)

def main():

    # Export to onnx
    model = Model(1)
    model.eval()
    device = torch.device("cpu")
    onnx_name = "argmax.onnx"
    dummy_input = torch.randn((3, 4), device=device)
    torch.onnx.export(model, dummy_input, onnx_name,
                      verbose=False, opset_version=16)
    
    print("Finished exporting model to {}".format(onnx_name))

    # Output some test data for use in the test
    test_input = torch.randn((2, 3), device=device)
    print("Test input data shape: {}".format(test_input.shape))
    output = model.forward(test_input)

    print("Test output data shape: {}".format(output.shape))



if __name__ == '__main__':
    main()
