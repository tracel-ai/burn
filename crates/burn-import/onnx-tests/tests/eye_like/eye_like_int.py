#!/usr/bin/env python3

# Used to generate model: onnx-tests/tests/eye_like/eye_like_int.onnx

import numpy as np
import onnx
from onnx import helper, TensorProto

def main():
    # Create ONNX model with EyeLike operator with integer output type
    onnx_file = "eye_like_int.onnx"

    # Define input and output
    input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, [3, 3])
    output_tensor = helper.make_tensor_value_info('output', TensorProto.INT64, [3, 3])

    # Create EyeLike node with dtype=INT64
    eye_like_node = helper.make_node(
        'EyeLike',
        inputs=['input'],
        outputs=['output'],
        name='EyeLike_int',
        dtype=TensorProto.INT64
    )

    # Create graph
    graph = helper.make_graph(
        [eye_like_node],
        'EyeLikeIntModel',
        [input_tensor],
        [output_tensor]
    )

    # Create model with proper opset specification
    eye_like_model = helper.make_model(
        graph,
        producer_name='eye_like_int_test',
        opset_imports=[helper.make_opsetid("", 16)]
    )

    # Save the model
    onnx.save(eye_like_model, onnx_file)

    print(f"Finished exporting model to {onnx_file}")

    # Expected output as integers
    expected_output = np.eye(3, dtype=np.int64)
    print(f"Expected output (INT64):\n{expected_output}")


if __name__ == '__main__':
    main()