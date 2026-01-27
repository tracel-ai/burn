#!/usr/bin/env python3

# used to generate model: maxpool2d_ceil_mode.onnx
# Tests ceil_mode=True for MaxPool2d which produces larger output when input doesn't divide evenly

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # Using 3x3 kernel with stride 2x2 on 6x6 input:
        # With ceil_mode=False: output = (6-3)/2+1 = 2x2
        # With ceil_mode=True: output = ceil((6-3)/2)+1 = 3x3
        self.maxpool2d = nn.MaxPool2d(
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=(0, 0),
            dilation=(1, 1),
            ceil_mode=True
        )

    def forward(self, x):
        x = self.maxpool2d(x)
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

    file_name = "maxpool2d_ceil_mode.onnx"
    # 6x6 input - doesn't divide evenly by stride 2 with kernel 3
    test_input = torch.tensor([[[[
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        [13.0, 14.0, 15.0, 16.0, 17.0, 18.0],
        [19.0, 20.0, 21.0, 22.0, 23.0, 24.0],
        [25.0, 26.0, 27.0, 28.0, 29.0, 30.0],
        [31.0, 32.0, 33.0, 34.0, 35.0, 36.0]
    ]]], device=device)

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
