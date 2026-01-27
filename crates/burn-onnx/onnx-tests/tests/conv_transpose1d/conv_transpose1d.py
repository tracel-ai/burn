#!/usr/bin/env python3

# used to generate model: conv_transpose1d.onnx

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.transposed_conv = nn.ConvTranspose1d(
            in_channels=4, 
            out_channels=6, 
            kernel_size=3, 
            stride=2, 
            padding=1, 
            dilation=2, 
            output_padding=1, 
            groups=2
        )

    def forward(self, x):
        return self.transposed_conv(x)


def main():

    # Set seed for reproducibility
    torch.manual_seed(42)

    torch.set_printoptions(precision=8)

    # Export to onnx
    model = Model()
    model.eval()
    device = torch.device("cpu")

    file_name = "conv_transpose1d.onnx"
    test_input = torch.ones(2, 4, 10, device=device)
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
