#!/usr/bin/env python3

"""
Script to generate an ONNX model with shape operations.
This generates the 'slice_shape.onnx' model that demonstrates
tensor shape slicing functionality.
"""

from typing import Text
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that extracts a slice of a tensor's shape.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input_tensor):
        # Extract tensor shape as a dynamic tensor (exports as ONNX Shape node)
        tensor_shape = torch._shape_as_tensor(input_tensor)
        # Extract dimensions 1 and 2 (exports as ONNX Slice node)
        shape_slice = tensor_shape[1:3]
        return shape_slice

def main():
    """Main function to create and export the ONNX model."""
    # Set deterministic seed for reproducible results
    torch.manual_seed(42)

    torch.set_printoptions(precision=8)

    # Initialize model in evaluation mode
    model = Model()
    model.eval()
    device = torch.device("cpu")

    output_filename = "slice_shape.onnx"

    # Create sample input tensor with dimensions [1, 2, 3, 1]
    sample_input = torch.randn(1, 2, 3, 1, device=device)

    # Export model to ONNX format
    torch.onnx.export(model, sample_input, output_filename,
                      input_names=['input'],
                      output_names=['output'],
                      dynamic_axes={'input': {0: 'input_dim'}, 'output': {0: 'output_dim'}},
                      verbose=False,
                      opset_version=16)

    print(f"Successfully exported model to {output_filename}")

    # Demonstrate model behavior with sample input
    print(f"Sample input tensor shape: {sample_input.shape}")
    model_output = model.forward(sample_input)
    print(f"Model output tensor: {model_output}")


if __name__ == '__main__':
    main()
