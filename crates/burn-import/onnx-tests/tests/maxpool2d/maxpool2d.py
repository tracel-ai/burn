#!/usr/bin/env python3

# used to generate model: maxpool2d1.onnx

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.maxpool2d1 = nn.MaxPool2d((4, 2), stride=(2, 1), padding=(2, 1), dilation=(1, 3))

    def forward(self, x):
        x = self.maxpool2d1(x)
        return x


def main():
    # Set seed for reproducibility
    torch.manual_seed(42)

    # Print options
    torch.set_printoptions(precision=3)

    # Export to onnx
    model = Model()
    model.eval()
    device = torch.device("cpu")

    file_name = "maxpool2d.onnx"
    test_input = torch.randn(1, 1, 5, 5, device=device)
    torch.onnx.export(model, test_input, file_name,
                      verbose=False, opset_version=16)

    print("Finished exporting model to {}".format(file_name))

    # Output some test data for use in the test
    print("Test input data shape of ones: {}".format(test_input.shape))
    print("Test input data of ones: {}".format(test_input))
    output = model.forward(test_input)
    print("Test output data shape: {}".format(output.shape))
    print("Test output: {}".format(output))


if __name__ == '__main__':
    main()
