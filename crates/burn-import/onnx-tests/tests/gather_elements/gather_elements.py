#!/usr/bin/env python3

# used to generate model: onnx-tests/tests/gather/gather_elements.onnx
# note that the ONNX specification for `GatherElements` corresponds to PyTorch's/Burn's `gather` function

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, index):
        x = torch.gather(x, 1, index)
        return x


def main():
    # Set random seed for reproducibility
    torch.manual_seed(0)

    # Export to onnx
    model = Model()
    model.eval()
    device = torch.device("cpu")
    onnx_name = "gather_elements.onnx"
    dummy_input = torch.randn(2, 2, device=device)
    dummy_index = torch.randint(high=2, size=(2, 2), device=device, dtype=torch.int64)

    torch.onnx.export(model, (dummy_input, dummy_index), onnx_name,
                      verbose=False, opset_version=16)

    print("Finished exporting model to {}".format(onnx_name))

    # Output some test data for use in the test
    test_input = torch.tensor([[1.0, 2.0],
                               [3.0, 4.0]])
    test_index = torch.tensor([[0, 0],
                               [1, 0]])

    print("Test input data: {}, {}".format(test_input, test_index))
    output = model.forward(test_input, test_index)
    print("Test output data: {}".format(output))


if __name__ == '__main__':
    main()