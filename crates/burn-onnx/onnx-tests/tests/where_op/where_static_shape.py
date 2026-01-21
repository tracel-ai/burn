#!/usr/bin/env python3

import torch
import torch.nn as nn
import onnx

class WhereStaticShapeModel(nn.Module):
    def forward(self, condition):
        # Create constant tensors with known static shapes
        # This tests that Where properly propagates static shapes from constant nodes
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)  # Static 2x2
        y = torch.tensor([[5.0, 6.0], [7.0, 8.0]], dtype=torch.float32)  # Static 2x2

        # Where should propagate the static shape [2, 2] from the constant tensors
        result = torch.where(condition, x, y)

        return result

def main():
    # Create model
    model = WhereStaticShapeModel()
    model.eval()

    # Create dummy input - boolean condition tensor
    dummy_condition = torch.tensor([[True, False], [False, True]], dtype=torch.bool)

    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_condition,
        "where_static_shape.onnx",
        input_names=["condition"],
        output_names=["output"],
        dynamic_axes={
            "condition": {0: "batch", 1: "features"},
        },
        opset_version=16,
    )

    print("Model exported to where_static_shape.onnx")

    # Test the model
    output = model(dummy_condition)
    print(f"Condition shape: {dummy_condition.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output:\n{output}")

    # Verify shape is propagated correctly
    assert output.shape == torch.Size([2, 2]), f"Expected shape [2, 2], got {output.shape}"

    # Verify values
    expected = torch.tensor([[1.0, 6.0], [7.0, 4.0]], dtype=torch.float32)
    assert torch.allclose(output, expected), f"Expected {expected}, got {output}"
    print("Test passed!")

if __name__ == "__main__":
    main()
