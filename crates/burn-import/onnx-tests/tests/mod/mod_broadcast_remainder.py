#!/usr/bin/env python3

# used to generate model: onnx-tests/tests/mod/mod_broadcast_remainder.onnx
# Tests broadcasting with fmod=0 (Python remainder)

import numpy as np
import onnx
from onnx import helper, TensorProto

def main():
    # Create ONNX model with Mod operator using broadcasting and remainder
    # Scalar broadcast scenario
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 4, 1])  # 3D
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [3, 1, 5])  # 3D

    # Output tensor will have the broadcasted shape
    z = helper.make_tensor_value_info('z', TensorProto.FLOAT, [3, 4, 5])

    # Create Mod node with fmod=0 (Python-style remainder)
    mod_node = helper.make_node(
        'Mod',
        inputs=['x', 'y'],
        outputs=['z'],
        fmod=0  # Python-style remainder
    )

    # Create the graph
    graph_def = helper.make_graph(
        [mod_node],
        'mod_broadcast_remainder_model',
        [x, y],
        [z],
    )

    # Create the model with opset version 16
    model_def = helper.make_model(
        graph_def,
        producer_name='onnx-tests',
        opset_imports=[helper.make_operatorsetid("", 16)]
    )

    # Save the model
    onnx_name = "mod_broadcast_remainder.onnx"
    onnx.save(model_def, onnx_name)
    print(f"Finished exporting model to {onnx_name}")

    # Test with NumPy to show expected results
    test_x = np.array([[[7.5], [-8.5], [9.5], [-10.5]]]).astype(np.float32)
    test_y = np.array([[[3.0, 4.0, -3.0, -4.0, 5.0]],
                       [[3.0, 4.0, -3.0, -4.0, 5.0]],
                       [[3.0, 4.0, -3.0, -4.0, 5.0]]]).astype(np.float32)

    # Broadcast both tensors and apply remainder
    result = np.remainder(test_x, test_y)

    print(f"Test input x shape: {test_x.shape}")
    print(f"Test input y shape: {test_y.shape}")
    print(f"Test output shape: {result.shape}")
    print(f"Sample x values: {test_x[0, :, 0]}")
    print(f"Sample y values: {test_y[0, 0, :]}")
    print(f"Sample output values: {result[0, 0, :]}  # First row result")

if __name__ == '__main__':
    main()