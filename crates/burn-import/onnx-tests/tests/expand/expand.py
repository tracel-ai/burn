#!/usr/bin/env python3

# used to generate model: onnx-tests/tests/expand/expand.onnx

import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, *y):
        return x.expand(*y)

def main():
    # Set seed for reproducibility
    torch.manual_seed(42)

    # Export to onnx
    model = Model()
    model.eval()
    device = torch.device("cpu")

    onnx_name = "expand.onnx"

    test_input1 = torch.tensor([[1], [2], [3]])
    test_input2 = (3,4)
    torch.onnx.export(model, (test_input1, test_input2), onnx_name, verbose=False, opset_version=16)

    print("Finished exporting model to {}".format(onnx_name))

    print("Test input data: {} (3,4)".format(test_input1))
    output = model.forward(test_input1, test_input2)
    print("Test output data: {}".format(output))
    # Output should be:
    # tensor([[ 1,  1,  1,  1],
    #     [ 2,  2,  2,  2],
    #     [ 3,  3,  3,  3]])

if __name__ == '__main__':
    main()
