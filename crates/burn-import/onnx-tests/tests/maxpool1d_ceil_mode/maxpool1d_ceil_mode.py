#!/usr/bin/env python3

# used to generate model: maxpool1d_ceil_mode.onnx
# Tests ceil_mode=True for MaxPool1d which produces larger output when input doesn't divide evenly

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # Using kernel 3 with stride 2 on length 6 input:
        # With ceil_mode=False: output = (6-3)/2+1 = 2
        # With ceil_mode=True: output = ceil((6-3)/2)+1 = 3
        self.maxpool1d = nn.MaxPool1d(
            kernel_size=3,
            stride=2,
            padding=0,
            dilation=1,
            ceil_mode=True
        )

    def forward(self, x):
        x = self.maxpool1d(x)
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

    file_name = "maxpool1d_ceil_mode.onnx"
    # 1x1x6 input with values 1-6 - doesn't divide evenly by stride 2 with kernel 3
    test_input = torch.tensor([[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]], device=device)

    torch.onnx.export(model, test_input, file_name,
                      verbose=False, opset_version=16)

    print("Finished exporting model to {}".format(file_name))

    # Output some test data for use in the test
    print("Test input data shape: {}".format(test_input.shape))
    print("Test input data: {}".format(test_input))
    output = model.forward(test_input)
    print("Test output data shape: {}".format(output.shape))
    print("Test output: {}".format(output))


if __name__ == '__main__':
    main()
