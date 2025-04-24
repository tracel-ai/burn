#!/usr/bin/env python3

# used to generate model: unsqueeze.onnx

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.axis = 3

    def forward(self, x, scalar):
        x = torch.unsqueeze(x, self.axis)
        y = torch.unsqueeze(torch.tensor(scalar), 0)
        return x, y


def main():
    # Set seed for reproducibility
    torch.manual_seed(42)

    torch.set_printoptions(precision=8)

    # Export to onnx
    model = Model()
    model.eval()
    device = torch.device("cpu")

    test_input = (torch.randn(3, 4, 5, device=device),1.0)
    model = Model()

    output = model.forward(*test_input)

    torch.onnx.export(model, test_input, "unsqueeze_like.onnx", verbose=False, opset_version=16)

    print(f"Finished exporting model")

    # Output some test data for use in the test
    print(f"Test input data shape of ones: {test_input[0].shape}")
    print(f"Test output data shape: {output[0].shape}")



if __name__ == "__main__":
    main()
