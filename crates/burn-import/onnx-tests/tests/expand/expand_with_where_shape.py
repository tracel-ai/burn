#!/usr/bin/env python3

import torch
import torch.nn as nn
import numpy as np
import onnx

class ExpandWithWhereShapeModel(nn.Module):
    def forward(self, x, condition):
        # Create a scenario similar to xfeat model where:
        # 1. Where operation produces a Shape through constant propagation
        # 2. That Shape is used as input to Expand

        # Create constant shape tensors
        shape1 = torch.tensor([2, 3, 4], dtype=torch.int64)
        shape2 = torch.tensor([2, 3, 4], dtype=torch.int64)

        # Use Where to select shape - this should produce a static shape
        selected_shape = torch.where(condition, shape1, shape2)

        # Use the result as shape for Expand
        # This tests that Expand can determine rank from Where's static shape output
        result = x.expand(selected_shape.tolist())

        return result

def main():
    # Create model
    model = ExpandWithWhereShapeModel()
    model.eval()

    # Create dummy inputs
    dummy_x = torch.ones(1, 1, 4)  # Broadcastable to [2, 3, 4]
    dummy_condition = torch.tensor([True, False, True], dtype=torch.bool)

    # Export to ONNX
    torch.onnx.export(
        model,
        (dummy_x, dummy_condition),
        "expand_with_where_shape.onnx",
        input_names=["input", "condition"],
        output_names=["output"],
        dynamic_axes={
            "input": {2: "dim2"},
            "condition": {0: "cond_dim"},
        },
        opset_version=17,
    )

    print("Model exported to expand_with_where_shape.onnx")

    # Test the model
    output = model(dummy_x, dummy_condition)
    print(f"Input shape: {dummy_x.shape}")
    print(f"Condition shape: {dummy_condition.shape}")
    print(f"Output shape: {output.shape}")

    # Verify output shape
    assert output.shape == torch.Size([2, 3, 4]), f"Expected shape [2, 3, 4], got {output.shape}"
    print("Test passed!")

if __name__ == "__main__":
    main()