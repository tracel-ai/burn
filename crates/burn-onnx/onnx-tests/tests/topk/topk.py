#!/usr/bin/env python3

import torch
import torch.nn as nn


class TopKModel(nn.Module):
    def __init__(self, k=1, dim=-1, largest=True, sorted=True):
        super(TopKModel, self).__init__()
        self.k = k
        self.dim = dim
        self.largest = largest
        self.sorted = sorted

    def forward(self, x):
        values, indices = torch.topk(
            x,
            k=self.k,
            dim=self.dim,
            largest=self.largest,
            sorted=self.sorted
        )
        return values, indices


def main():
    # Set seed for reproducibility
    torch.manual_seed(42)

    # Set print options for better precision output
    torch.set_printoptions(precision=8)

    # Export TopK Model
    k = 2  # Number of top elements to return
    dim = 1  # Dimension along which to find top k elements
    largest = True  # Whether to return largest or smallest elements
    sorted = True  # Whether to return the elements in sorted order

    model = TopKModel(k=k, dim=dim, largest=largest, sorted=sorted)
    model.eval()
    device = torch.device("cpu")

    # Generate test input
    file_name = "topk.onnx"
    test_input = torch.randn(3, 5, device=device)  # 3 sequences of 5 elements
    torch.onnx.export(model, test_input, file_name,
                      verbose=False, opset_version=16)

    print("Finished exporting model to {}".format(file_name))

    # Output some test data for use in the test
    print("Test input data: {}".format(test_input))
    print("Test input data shape: {}".format(test_input.shape))
    values, indices = model.forward(test_input)
    print("Test output values shape: {}".format(values.shape))
    print("Test output values: {}".format(values))
    print("Test output indices shape: {}".format(indices.shape))
    print("Test output indices: {}".format(indices))

if __name__ == '__main__':
    main()
