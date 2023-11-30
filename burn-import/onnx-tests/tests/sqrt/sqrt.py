#!/usr/bin/env python3

# used to generate model: onnx-tests/tests/sqrt/sqrt.onnx

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y):
        y_tensor = torch.tensor(y)  # Convert y to a PyTorch tensor
        return torch.sqrt(x), torch.sqrt(y_tensor)

def main():
    # Set random seed for reproducibility
    torch.manual_seed(0)

    # Export to onnx
    model = Model()
    model.eval()
    device = torch.device("cpu")
    onnx_name = "sqrt.onnx"
    test_input1 = torch.tensor([[[[1.0, 4.0, 9.0, 25.0]]]])
    test_input2 = 36.0
    torch.onnx.export(model, (test_input1, test_input2), onnx_name,
                      verbose=False, opset_version=16)

    print("Finished exporting model to {}".format(onnx_name))

    # Output some test data for use in the test
    test_input1 = torch.tensor([[[[1.0, 4.0, 9.0, 25.0]]]])
    test_input2 = 36.0

    print("Test input data: {}, {}".format(test_input1, test_input2))
    output1, output2 = model.forward(test_input1, test_input2)
    print("Test output data: {}, {}".format(output1, output2))


if __name__ == '__main__':
    main()
