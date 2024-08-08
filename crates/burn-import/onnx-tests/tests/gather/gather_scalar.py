#!/usr/bin/env python3

# used to generate model: onnx-tests/tests/gather/gather_scalar.onnx

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, index):
        gathered = torch.select(x, 0, index)
        return gathered


def main():
    # Set random seed for reproducibility
    torch.manual_seed(0)

    # Export to onnx
    model = Model()
    model.eval()
    device = torch.device("cpu")
    onnx_name = "gather_scalar.onnx"

    dummy_input = torch.randn(2, 3, device=device)
    dummy_index = 0

    torch.onnx.export(model, (dummy_input, dummy_index), onnx_name,
                      verbose=False, opset_version=16)

    print("Finished exporting model to {}".format(onnx_name))

    # Output some test data for use in the test
    test_input = torch.tensor([[1.0, 2.0, 3.0],
                               [4.0, 5.0, 6.0]])
    test_index = 0

    print("Test input data: {}, {}".format(test_input, test_index))
    output = model.forward(test_input, test_index)
    print("Test output data: {}".format(output))


if __name__ == '__main__':
    main()
