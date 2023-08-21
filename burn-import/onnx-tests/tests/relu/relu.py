#!/usr/bin/env python3

# used to generate model: relu.onnx

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.relu1 = nn.ReLU()

    def forward(self, x):
        x = self.relu1(x)
        return x


def main():

    # Set seed for reproducibility
    torch.manual_seed(42)

    torch.set_printoptions(precision=8)
                           
    # Export to onnx
    model = Model()
    model.eval()
    device = torch.device("cpu")

    file_name = "relu.onnx"
    test_input = torch.randn(2, 3, device=device)
    torch.onnx.export(model, test_input, file_name,
                      verbose=False, opset_version=16)

    print("Finished exporting model to {}".format(file_name))

    # Output some test data for use in the test
    print("Test input data of ones: {}".format(test_input))
    print("Test input data shape of ones: {}".format(test_input.shape))
    output = model.forward(test_input)
    print("Test output data shape: {}".format(output.shape))

    print("Test output: {}".format(output))


if __name__ == '__main__':
    main()
