#!/usr/bin/env python3

# used to generate model: clip_opset16.onnx

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):
        x1 = x.clamp(min=0.3)
        x2 = x.clamp(min=0.5, max=0.7)
        x3 = x.clamp(max=0.8)
        return x1, x2, x3


def main():

    # Set seed for reproducibility
    torch.manual_seed(42)

    torch.set_printoptions(precision=8)

    # Export to onnx
    model = Model()
    model.eval()
    device = torch.device("cpu")

    file_name = "clip.onnx"
    test_input = torch.rand(6, device=device)
    torch.onnx.export(model, test_input, file_name,
                      verbose=False, opset_version=16)

    print("Finished exporting model to {}".format(file_name))

    # Output some test data for use in the test
    print("Test input data: {}".format(test_input))
    print("Test input data shape: {}".format(test_input.shape))
    x1, x2, x3 = model.forward(test_input)
    print("Test output data shape: {}, {}, {}".format(
        x1.shape, x2.shape, x3.shape))

    print("Test output: {}, {}, {}".format(x1, x2, x3))


if __name__ == '__main__':
    main()
