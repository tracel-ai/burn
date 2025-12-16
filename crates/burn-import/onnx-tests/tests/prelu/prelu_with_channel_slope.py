#!/usr/bin/env python3

# used to generate model: prelu_with_channel_slope.onnx

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.prelu = nn.PReLU(num_parameters=3)

    def forward(self, x):
        x = self.prelu(x)
        return x


def main():
    # Set seed for reproducibility
    torch.manual_seed(42)

    torch.set_printoptions(precision=8)

    # Export to onnx
    model = Model()
    model.eval()
    device = torch.device("cpu")

    file_name = "prelu_with_channel_slope.onnx"
    export_input = torch.randn(1, 3, 2, 2, device=device)
    torch.onnx.export(model, export_input, file_name,
                      verbose=False, opset_version=18, external_data=False)

    print("Finished exporting model to {}".format(file_name))

    test_input = torch.tensor([[
        [[0.5, -0.5], [1.0, -1.0]],    # ch0: mix of pos/neg
        [[0.1, 0.2], [0.3, 0.4]],      # ch1: all positive
        [[-0.1, -0.2], [-0.3, -0.4]],  # ch2: all negative
    ]], device=device)

    output = model.forward(test_input)

    print("\n// Test data:")
    print("// Input: {}".format(test_input.tolist()))
    print("// Output: {}".format(output.tolist()))
    print("// Slopes: {}".format(model.prelu.weight.tolist()))


if __name__ == '__main__':
    main()
