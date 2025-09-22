#!/usr/bin/env python3

# used to generate model: onnx-tests/tests/mod/mod_remainder.onnx
# Tests fmod=0 (Python-style remainder)

import numpy as np
import onnx
from onnx import helper, TensorProto

def main():
    # Create ONNX model with Mod operator using fmod=0 (remainder)
    # Input tensors - 3D for consistency with other tests
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 1, 4])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 1, 4])

    # Output tensor
    z = helper.make_tensor_value_info('z', TensorProto.FLOAT, [1, 1, 4])

    # Create Mod node with fmod=0 (Python remainder %)
    mod_node = helper.make_node(
        'Mod',
        inputs=['x', 'y'],
        outputs=['z'],
        fmod=0  # Python-style remainder
    )

    # Create the graph
    graph_def = helper.make_graph(
        [mod_node],
        'mod_remainder_model',
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
    onnx_name = "mod_remainder.onnx"
    onnx.save(model_def, onnx_name)
    print(f"Finished exporting model to {onnx_name}")

    # Test with NumPy to show expected results
    test_x = np.array([[[[5.3, -5.3, 7.5, -7.5]]]]).astype(np.float32)
    test_y = np.array([[[[2.0, 2.0, 3.0, 3.0]]]]).astype(np.float32)

    # Python remainder: sign follows divisor
    result = np.remainder(test_x, test_y)

    print(f"Test input x: {test_x}")
    print(f"Test input y: {test_y}")
    print(f"Test output (remainder): {result}")
    print(f"Expected values: [1.3, 0.7, 1.5, 1.5] (sign follows divisor)")

if __name__ == '__main__':
    main()