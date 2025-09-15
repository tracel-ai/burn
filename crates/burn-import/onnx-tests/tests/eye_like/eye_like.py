#!/usr/bin/env python3

# Used to generate model: onnx-tests/tests/eye_like/eye_like.onnx

import numpy as np
import onnx
from onnx import helper, TensorProto

def main():
    # Create ONNX model with EyeLike operator directly
    onnx_file = "eye_like.onnx"

    # Define input and output with dynamic shape (can handle different matrix sizes)
    input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, ['H', 'W'])
    output_tensor = helper.make_tensor_value_info('output', TensorProto.FLOAT, ['H', 'W'])

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

    # Test input data examples (values don't matter for EyeLike, only shape)
    print("EyeLike operator test cases:")

    # Test case 1: 3x3 square matrix
    test_input_3x3 = np.zeros((3, 3), dtype=np.float32)
    expected_output_3x3 = np.eye(3, dtype=np.float32)
    print(f"Square matrix (3x3) input shape: {test_input_3x3.shape}")
    print(f"Expected output:\n{expected_output_3x3}")

    # Test case 2: 3x4 rectangular matrix
    test_input_3x4 = np.zeros((3, 4), dtype=np.float32)
    expected_output_3x4 = np.zeros((3, 4), dtype=np.float32)
    expected_output_3x4[:3, :3] = np.eye(3)  # Identity in top-left corner
    print(f"Rectangular matrix (3x4) input shape: {test_input_3x4.shape}")
    print(f"Expected output:\n{expected_output_3x4}")


if __name__ == '__main__':
    main()