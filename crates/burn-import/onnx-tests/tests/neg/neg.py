#!/usr/bin/env python3

# used to generate model: onnx-tests/tests/neg/neg.onnx

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y):
        return torch.neg(x), -y


def main():
    # Set random seed for reproducibility
    torch.manual_seed(0)

    # Export to onnx
    model = Model()
    model.eval()
    device = torch.device("cpu")
    onnx_name = "neg.onnx"
    test_input1 = torch.tensor([[[[1.0, 4.0, 9.0, 25.0]]]], device=device)
    test_input2 = 99.0

    torch.onnx.export(model, (test_input1, test_input2), onnx_name,
                      verbose=False, opset_version=16)

    print("Finished exporting model to {}".format(onnx_name))

    # Output some test data for use in the test
    print("Test input1: {}, input2: {}".format(test_input1, test_input2))
    output1, output2 = model.forward(test_input1, test_input2)
    print("Test output1 data: {}".format(output1))
    print("Test output2 data: {}".format(output2))


if __name__ == '__main__':
    main()
