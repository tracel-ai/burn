#!/usr/bin/env python3

# used to generate model: onnx-tests/tests/conv1d/conv1d.onnx

# TODO finish this

import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv1d(4, 6, (3, 5), groups = 2, stride=1, padding=2, dilation=1)

    def forward(self, x):
        x = self.conv1(x)
        return x

def main():

    # Export to onnx
    model = Model()
    model.eval()
    device = torch.device("cpu")

    file_name = "conv1d.onnx"
    test_input = torch.ones(2, 4, 10, 15, device=device)
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
