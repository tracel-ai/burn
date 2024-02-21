#!/usr/bin/env python3

# used to generate model: onnx-tests/tests/conv2d/conv2d.onnx

import torch
import torch.nn as nn

# must set for testing against crate
torch.manual_seed(0)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(
            4, 6, (3, 5), groups=2, stride=(2, 1), padding=(4, 2), dilation=(3, 1)
        )

    def forward(self, x):
        x = self.conv1(x)
        return x


def main():

    # Set random seed for reproducibility
    torch.manual_seed(0)

    # Export to onnx
    model = Model()
    model.eval()
    device = torch.device("cpu")

    file_name = "conv2d.onnx"
    test_input = torch.ones(2, 4, 10, 15, device=device)
    torch.onnx.export(model, test_input, file_name, verbose=False, opset_version=16)

    print("Finished exporting model to {}".format(file_name))

    # Output some test data for use in the test
    print("Test input data shape of ones: {}".format(test_input.shape))
    output = model.forward(test_input)
    print("Test output data shape: {}".format(output.shape))

    sum = output.sum().item()

    print("Test output sum: {}".format(sum))


if __name__ == "__main__":
    main()
