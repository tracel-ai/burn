#!/usr/bin/env python3

# used to generate model: global_avr_pool.onnx

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.pool1 = nn.AdaptiveAvgPool1d(1)
        self.pool2 = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x_1d, x_2d):
        y_1d = self.pool1(x_1d)
        y_2d = self.pool2(x_2d)
        return y_1d, y_2d


def main():

    # Export to onnx
    model = Model()
    model.eval()
    device = torch.device("cpu")

    file_name = "global_avr_pool.onnx"
    input1 = torch.ones(2, 4, 10, device=device)
    input2 = torch.ones(3, 10, 3, 15, device=device)
    torch.onnx.export(model, (input1, input2), file_name,
                      verbose=False, opset_version=16)

    print("Finished exporting model to {}".format(file_name))

    # Output some test data for use in the test
    print("Test input data shapes of ones: {}, {}".format(
        input1.shape, input2.shape))
    y_1d, y_2d = model.forward(input1, input2)
    print("Test output data shapes: {}, {}".format(y_1d.shape, y_2d.shape))

    sum1 = y_1d.sum().item()
    sum2 = y_2d.sum().item()

    print("Test output sums: {}, {}".format(sum1, sum2))


if __name__ == '__main__':
    main()
