#!/usr/bin/env python3

# used to generate model: split.onnx

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):
        x = torch.split(x, 2)
        return x


def main():
    # Set seed for reproducibility
    torch.manual_seed(42)

    torch.set_printoptions(precision=8)

    model = Model()
    model.eval()
    device = torch.device("cpu")

    file_name = "split.onnx"
    test_input = torch.arange(10, device=device).reshape(5, 2)
    torch.onnx.export(model, test_input, file_name,
                        verbose=False, opset_version=16)
    print("Finished exporting model to {}".format(file_name))

    print("Test input data shape: {}".format(test_input.shape))
    print("Splitting input tensor into chunks of size 2")
    output = model.forward(test_input)
    print("Test output data length: {}".format(len(output)))

if __name__ == '__main__':
    main()
