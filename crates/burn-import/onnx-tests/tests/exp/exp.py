#!/usr/bin/env python3

# used to generate model: onnx-tests/tests/exp/exp.onnx

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):
        return torch.exp(x)


def main():
    # Set random seed for reproducibility
    torch.manual_seed(0)
    import math
    
    # Export to onnx
    model = Model()
    model.eval()
    device = torch.device("cpu")
    onnx_name = "exp.onnx"
    test_input = torch.tensor([[[[0, math.log(2.)]]]], device=device)

    torch.onnx.export(model, (test_input), onnx_name,
                      verbose=False, opset_version=16)

    print("Finished exporting model to {}".format(onnx_name))

    # Output some test data for use in the test
    print("Test input data: {}".format(test_input))
    output = model.forward(test_input)
    print("Test output data: {}".format(output))


if __name__ == '__main__':
    main()
