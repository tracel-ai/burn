#!/usr/bin/env python3

# used to generate model: onnx-tests/tests/is_inf_scalar/is_inf_scalar.onnx

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):
        return torch.isinf(x)

def main():
    # Set seed for reproducibility
    torch.manual_seed(42)
    torch.set_printoptions(precision=8)

    # Export to onnx
    model = Model()
    model.eval()
    device = torch.device("cpu")

    onnx_name = "is_inf_scalar.onnx"

    test_input1 = torch.tensor(1.0, device=device)
    torch.onnx.export(model, (test_input1,), onnx_name, verbose=False, opset_version=16)

    print("Finished exporting model to {}".format(onnx_name))

    print("Test input data: {}".format(test_input1))
    output = model.forward(test_input1)
    print("Test output data: {}".format(output))

if __name__ == '__main__':
    main()
