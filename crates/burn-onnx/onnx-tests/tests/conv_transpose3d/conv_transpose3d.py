#!/usr/bin/env python3

# used to generate model: onnx-tests/tests/conv_transpose3d/conv_transpose3d.onnx

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.transposed_conv = nn.ConvTranspose3d(
            4, 6, (3, 5, 5), groups=2, stride=(2, 1, 1), padding=(4, 2, 1), dilation=(3, 1, 1), output_padding=(1, 0, 0),
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

    file_name = "conv_transpose3d.onnx"
    test_input = torch.ones(2, 4, 4, 5, 7, device=device)
    torch.onnx.export(model, test_input, file_name,
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
