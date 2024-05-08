#!/usr/bin/env python3

# used to generate model: onnx-tests/tests/cumsum/cumsum.onnx

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # self.b = 5.0

    def forward(self, x, d):
        # cumulative sum of a tensor along dimension d
        x = x.cumsum(d)

        return x


def main():
    # Export to onnx
    model = Model()
    model.eval()
    device = torch.device("cpu")
    onnx_name = "cumsum.onnx"
    dummy_input = torch.tensor([[0,1,2], [3,4,5], [6, 7, 8]], dtype=torch.float32, device=device)

    dim= 1

    torch.onnx.export(
        model, (dummy_input, dim), onnx_name, verbose=False, opset_version=16
    )

    print(f"Finished exporting model to {onnx_name}")

    # Output some test data for use in the test
    test_input = torch.tensor([[0,1,2], [3,4,5], [6, 7, 8]], dtype=torch.float32, device=device)

    print(f"Test input data: {test_input}, {dim}")
    output = model.forward(test_input, dim)
    print(f"Test output data: {output}")

    

if __name__ == "__main__":
    main()
