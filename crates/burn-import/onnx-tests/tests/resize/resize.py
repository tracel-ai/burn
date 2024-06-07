#!/usr/bin/env python3

# used to generate model: resize.onnx

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):
        x = F.interpolate(x, size=(2, 2), mode='bilinear', align_corners=True)
        return x


def main():

    # Set seed for reproducibility
    torch.manual_seed(42)

    torch.set_printoptions(precision=8)

    # Export to onnx
    model = Model()
    model.eval()

    file_name = "resize.onnx"
    test_input = torch.randn(1, 1, 4, 4)

    torch.onnx.export(
        model,
        test_input,
        file_name,
        verbose=False,
        opset_version=16
    )

    print("Finished exporting model to {}".format(file_name))

    # Output some test data for use in the test
    print("Test input data of ones: {}".format(test_input))
    print("Test input data shape of ones: {}".format(test_input.shape))
    output = model.forward(test_input)
    print("Test output data shape: {}".format(output.shape))

    print("Test output: {}".format(output))


if __name__ == '__main__':
    main()
