#!/usr/bin/env python3

# Used to generate model: onnx-tests/tests/eye_like/eye_like_k1.onnx

import numpy as np
import onnx
from onnx import helper, TensorProto

def main():
    # Create ONNX model with EyeLike operator with k=1 (upper diagonal)
    onnx_file = "eye_like_k1.onnx"

    # Define input and output
    input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, [4, 4])
    output_tensor = helper.make_tensor_value_info('output', TensorProto.FLOAT, [4, 4])

    # Create EyeLike node with k=1 (upper diagonal)
    eye_like_node = helper.make_node(
        'EyeLike',
        inputs=['input'],
        outputs=['output'],
        name='EyeLike_k1',
        k=1
    )

    # Create graph
    graph = helper.make_graph(
        [eye_like_node],
        'EyeLikeK1Model',
        [input_tensor],
        [output_tensor]
    )

    # Create model with proper opset specification
    eye_like_model = helper.make_model(
        graph,
        producer_name='eye_like_k1_test',
        opset_imports=[helper.make_opsetid("", 16)]
    )

    # Save the model
    onnx.save(eye_like_model, onnx_file)

    print(f"Finished exporting model to {onnx_file}")

    # Expected output with k=1 (upper diagonal)
    expected_output = np.array([
        [0., 1., 0., 0.],
        [0., 0., 1., 0.],
        [0., 0., 0., 1.],
        [0., 0., 0., 0.]
    ], dtype=np.float32)
    print(f"Expected output (k=1):\n{expected_output}")


if __name__ == '__main__':
    main()