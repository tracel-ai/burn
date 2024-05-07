#!/usr/bin/env python3

# used to generate model: avg_pool1d.onnx

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.pool1 = nn.AvgPool1d(4, stride=2)

        self.pool2 = nn.AvgPool1d(4, stride=2, padding=2, count_include_pad=True)

        self.pool3 = nn.AvgPool1d(4, stride=2, padding=2, count_include_pad=False)

    def forward(self, x1, x2, x3):
        y1 = self.pool1(x1)
        y2 = self.pool2(x2)
        y3 = self.pool3(x3)
        return y1, y2, y3


def main():
    # Set seed for reproducibility
    torch.manual_seed(1)

    # Print options
    torch.set_printoptions(precision=3)

    # Export to onnx
    model = Model()
    model.eval()
    device = torch.device("cpu")

    file_name = "avg_pool1d.onnx"
    input1 = torch.randn(1, 5, 5, device=device)
    torch.onnx.export(model, (input1, input1, input1), file_name,
                      verbose=False, opset_version=16)

    print("Finished exporting model to {}".format(file_name))

    # Output some test data for use in the test
    print("Test input data shape: {}".format(input1.shape))
    print("Test input data: {}".format(input1))
    output1, output2, output3 = model.forward(input1, input1, input1)
    print("Test output1 data shape: {}".format(output1.shape))
    print("Test output2 data shape: {}".format(output2.shape))
    print("Test output3 data shape: {}".format(output3.shape))
    print("Test output1: {}".format(output1))
    print("Test output2: {}".format(output2))
    print("Test output3: {}".format(output3))


if __name__ == '__main__':
    main()
