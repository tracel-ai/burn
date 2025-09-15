#!/usr/bin/env python3

# Used to generate model: onnx-tests/tests/eye_like/eye_like.onnx

import torch
import torch.nn as nn
import onnx

class EyeLikeModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # EyeLike is not directly available in PyTorch, but we can use onnx directly
        # We'll create a simple identity matrix operation
        # For the purpose of this test, we'll simulate it
        batch_size = x.shape[0] if x.dim() > 2 else 1
        height, width = x.shape[-2], x.shape[-1]

        # Create identity matrix
        eye = torch.eye(min(height, width), dtype=x.dtype, device=x.device)

        # Pad to match input shape if needed
        if height != width:
            result = torch.zeros_like(x)
            min_dim = min(height, width)
            result[..., :min_dim, :min_dim] = eye
            return result
        else:
            return eye.expand_as(x)

def main():
    model = EyeLikeModel()
    model.eval()

    # Test with a 3x3 square matrix
    test_input = torch.zeros(3, 3)

    onnx_file = "eye_like.onnx"

    # We need to create the ONNX model manually since PyTorch doesn't have EyeLike
    # Let's use a simpler approach with torch.onnx
    torch.onnx.export(
        model,
        test_input,
        onnx_file,
        opset_version=16,
    )

    # However, the exported model won't have the actual EyeLike op
    # We need to modify the ONNX model to use EyeLike operator

    # Load and modify the ONNX model
    model_proto = onnx.load(onnx_file)

    # Create a new ONNX model with EyeLike operator
    from onnx import helper, TensorProto, ValueInfoProto

    # Define input and output
    input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, [3, 3])
    output_tensor = helper.make_tensor_value_info('output', TensorProto.FLOAT, [3, 3])

    # Create EyeLike node
    eye_like_node = helper.make_node(
        'EyeLike',
        inputs=['input'],
        outputs=['output'],
        name='EyeLike_0'
    )

    # Create graph
    graph = helper.make_graph(
        [eye_like_node],
        'EyeLikeModel',
        [input_tensor],
        [output_tensor]
    )

    # Create model
    eye_like_model = helper.make_model(graph, producer_name='eye_like_test')
    eye_like_model.opset_import[0].version = 16

    # Save the model
    onnx.save(eye_like_model, onnx_file)

    print(f"Finished exporting model to {onnx_file}")
    print(f"Test input data shape: {test_input.shape}")

    # Expected output is identity matrix
    expected_output = torch.eye(3)
    print(f"Expected output: {expected_output}")


if __name__ == '__main__':
    main()