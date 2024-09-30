#!/usr/bin/env python3

import torch
import torch.nn as nn


class TriluModel(nn.Module):
    def __init__(self, upper=True, k=0):
        super(TriluModel, self).__init__()
        self.upper = upper  # Determines upper or lower triangular
        self.k = k  # Diagonal offset

    def forward(self, x):
        # torch.tril or torch.triu based on 'upper' attribute
        if self.upper:
            return torch.triu(x, diagonal=self.k)
        else:
            return torch.tril(x, diagonal=self.k)


def main():
    # Set seed for reproducibility
    torch.manual_seed(42)

    # Set print options for better precision output
    torch.set_printoptions(precision=8)
                           
    # Export to onnx
    upper = True  # Change to False for lower triangular matrix
    k = 0         # Change k to adjust the diagonal
    model = TriluModel(upper=upper, k=k)
    model.eval()
    device = torch.device("cpu")

    # Generate test input: a 2D matrix or batch of 2D matrices
    file_name = "trilu.onnx"
    test_input = torch.randn(2, 4, 4, device=device)  # 2 batches of 4x4 matrices
    torch.onnx.export(model, test_input, file_name,
                      verbose=False, opset_version=16)

    print("Finished exporting model to {}".format(file_name))

    # Output some test data for use in the test
    print("Test input data: {}".format(test_input))
    print("Test input data shape: {}".format(test_input.shape))
    output = model.forward(test_input)
    print("Test output data shape: {}".format(output.shape))

    print("Test output: {}".format(output))


if __name__ == '__main__':
    main()