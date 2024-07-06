#!/usr/bin/env python3
import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
import numpy as np

class Model(nn.Module):
    def __init__(self, fill_value):
        super(Model, self).__init__()
        self.fill_value = fill_value

    def forward(self, x):
        # Use full_like, which will be exported as ConstantOfShape
        return torch.full_like(x, self.fill_value)


def main():
    # Set random seed for reproducibility
    torch.manual_seed(0)

    # Create an instance of the model
    model = Model(fill_value=3.0)

    # Create a dummy input
    test_input = torch.randn(2, 3, 4)

    file_name = "constant_of_shape_full_like.onnx"

    # Export the model to ONNX
    torch.onnx.export(model, test_input, file_name,
                    verbose=False, opset_version=16,
                    input_names=['input'],
                    output_names=['output'],
                    dynamic_axes={'input': {0: 'batch_size', 1: 'height', 2: 'width'}})

    print("Finished exporting model to {}".format(file_name))

    # Output some test data for use in the test
    print("Test input data shape of ones: {}".format(test_input.shape))
    output = model.forward(test_input)
    print("Test output data shape: {}".format(output.shape))

    sum = output.sum().item()

    print("Test output sum: {}".format(sum))


if __name__ == "__main__":
    main()
