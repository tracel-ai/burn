#!/usr/bin/env python3

# used to generate model: linear.onnx

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        # When input's 2d with [1, n] shape and bias is True, Gemm ONNX node is used
        self.fc1 = nn.Linear(16, 32, bias=True)

        # TODO Test other cases that use matmul instead of Gemm

    def forward(self, x):
        x = self.fc1(x)
        return x


def main():

    # Set random seed for reproducibility
    torch.manual_seed(0)

    print("Making model")
    # Export to onnx
    model = Model()
    model.eval()
    device = torch.device("cpu")
    print("Made model")

    file_name = "linear.onnx"
    test_input = torch.full((1, 16), 3.141592, device=device)
    torch.onnx.export(model, test_input, file_name,
                      verbose=False, opset_version=16)

    print("Finished exporting model to {}".format(file_name))

    # Output some test data for use in the test
    print("Test input data shape of ones: {}".format(test_input.shape))
    output = model.forward(test_input)
    print("Test output data shape: {}".format(output.shape))

    sum = output.sum().item()

    print("Test output sum: {}".format(sum))


if __name__ == "__main__":
    main()
