#!/usr/bin/env python3

# used to generate model: batch_norm.onnx

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.batch_norm1d = nn.BatchNorm1d(20)
        self.batch_norm2d = nn.BatchNorm2d(5)

    def forward(self, x):
        x = self.batch_norm1d(x)
        x = x.reshape(1, 5, 2, 2)
        x = self.batch_norm2d(x)
        return x


def main():

    # Export to onnx
    model = Model()
    model.eval()
    device = torch.device("cpu")

    # reproducibility
    torch.manual_seed(0)

    file_name = "batch_norm.onnx"
    test_input = torch.ones(1, 20, 1, device=device)
    torch.onnx.export(model, test_input, file_name,
                    #   do_constant_folding=False,
                      verbose=False, opset_version=16)

    print("Finished exporting model to {}".format(file_name))

    # Output some test data for use in the test
    print("Test input data shape of ones: {}".format(test_input.shape))
    output = model.forward(test_input)
    print("Test output data shape: {}".format(output.shape))

    sum = output.sum().item()

    print("Test output sum: {}".format(sum))


if __name__ == '__main__':
    main()
