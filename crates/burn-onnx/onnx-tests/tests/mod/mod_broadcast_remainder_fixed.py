#!/usr/bin/env python3

# used to generate model: onnx-tests/tests/mod/mod_broadcast_remainder_fixed.onnx
# Tests broadcasting with remainder (Python %) - using torch for proper shape inference

import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y):
        # Use torch.remainder for Python-style % behavior
        return torch.remainder(x, y)

def main():
    # Export to onnx
    model = Model()
    model.eval()
    device = torch.device("cpu")
    onnx_name = "mod_broadcast_remainder_fixed.onnx"

    # Create dummy inputs with different shapes for broadcasting
    dummy_x = torch.randn(1, 4, 1, device=device)  # 3D tensor
    dummy_y = torch.randn(3, 1, 5, device=device)  # 3D tensor

    torch.onnx.export(model, (dummy_x, dummy_y), onnx_name,
                      verbose=False, opset_version=16)

    print(f"Finished exporting model to {onnx_name}")

    # Test with specific values
    test_x = torch.tensor([[[7.5], [-8.5], [9.5], [-10.5]]], dtype=torch.float32)
    test_y = torch.tensor([[[3.0, 4.0, -3.0, -4.0, 5.0]],
                           [[3.0, 4.0, -3.0, -4.0, 5.0]],
                           [[3.0, 4.0, -3.0, -4.0, 5.0]]], dtype=torch.float32)

    output = model.forward(test_x, test_y)
    print(f"Test input x shape: {test_x.shape}")
    print(f"Test input y shape: {test_y.shape}")
    print(f"Test output shape: {output.shape}")
    print(f"Sample output values: {output[0, 0, :]}  # First row result")

if __name__ == '__main__':
    main()