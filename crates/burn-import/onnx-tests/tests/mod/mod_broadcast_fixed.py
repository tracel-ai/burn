#!/usr/bin/env python3

# used to generate model: onnx-tests/tests/mod/mod_broadcast_fixed.onnx
# Tests broadcasting with fmod=1 - using torch for proper shape inference

import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y):
        # Use torch.fmod for C-style fmod behavior
        return torch.fmod(x, y)

def main():
    # Export to onnx
    model = Model()
    model.eval()
    device = torch.device("cpu")
    onnx_name = "mod_broadcast_fixed.onnx"

    # Create dummy inputs with different ranks for broadcasting
    dummy_x = torch.randn(3, 4, device=device)  # 2D tensor
    dummy_y = torch.randn(2, 1, 3, 4, device=device)  # 4D tensor

    torch.onnx.export(model, (dummy_x, dummy_y), onnx_name,
                      verbose=False, opset_version=16)

    print(f"Finished exporting model to {onnx_name}")

    # Test with specific values
    test_x = torch.tensor([[5.0, -7.0, 8.0, -9.0],
                           [4.0, -6.0, 10.0, -11.0],
                           [3.0, -5.0, 12.0, -13.0]], dtype=torch.float32)

    test_y = torch.tensor([[[[3.0, 3.0, 3.0, 3.0],
                             [3.0, 3.0, 3.0, 3.0],
                             [3.0, 3.0, 3.0, 3.0]]],
                           [[[4.0, 4.0, 4.0, 4.0],
                             [4.0, 4.0, 4.0, 4.0],
                             [4.0, 4.0, 4.0, 4.0]]]], dtype=torch.float32)

    output = model.forward(test_x, test_y)
    print(f"Test input x shape: {test_x.shape}")
    print(f"Test input y shape: {test_y.shape}")
    print(f"Test output shape: {output.shape}")
    print(f"Sample output values: {output[0, 0, 0, :]}  # First row, first batch")

if __name__ == '__main__':
    main()