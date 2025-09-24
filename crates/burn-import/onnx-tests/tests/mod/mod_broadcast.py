#!/usr/bin/env python3

# used to generate model: onnx-tests/tests/mod/mod_broadcast.onnx
# Tests broadcasting with fmod=1

import numpy as np
import onnx
from onnx import helper, TensorProto

def main():
    # Create ONNX model with Mod operator using broadcasting
    # Different rank tensors: 2D and 4D
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [3, 4])  # 2D
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [2, 1, 3, 4])  # 4D

    # Output tensor will have the broadcasted shape
    z = helper.make_tensor_value_info('z', TensorProto.FLOAT, [2, 1, 3, 4])

    # Create Mod node with fmod=1 (C-style fmod)
    mod_node = helper.make_node(
        'Mod',
        inputs=['x', 'y'],
        outputs=['z'],
        fmod=1  # C-style fmod
    )

    # Create the graph
    graph_def = helper.make_graph(
        [mod_node],
        'mod_broadcast_model',
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
    onnx_name = "mod_broadcast.onnx"
    onnx.save(model_def, onnx_name)
    onnx.checker.check_model(onnx_name)
    print(f"Finished exporting model to {onnx_name}")

    # Test with onnx.reference.ReferenceEvaluator
    try:
        from onnx.reference import ReferenceEvaluator

        # Create test data
        test_x = np.array([[5.0, -7.0, 8.0, -9.0],
                           [4.0, -6.0, 10.0, -11.0],
                           [3.0, -5.0, 12.0, -13.0]]).astype(np.float32)

        test_y = np.array([[[[3.0, 3.0, 3.0, 3.0],
                             [3.0, 3.0, 3.0, 3.0],
                             [3.0, 3.0, 3.0, 3.0]]],
                           [[[4.0, 4.0, 4.0, 4.0],
                             [4.0, 4.0, 4.0, 4.0],
                             [4.0, 4.0, 4.0, 4.0]]]]).astype(np.float32)

        # Run inference with ReferenceEvaluator
        sess = ReferenceEvaluator(model_def)
        result = sess.run(None, {"x": test_x, "y": test_y})

        print(f"Test input x shape: {test_x.shape}")
        print(f"Test input y shape: {test_y.shape}")
        print(f"Result shape: {result[0].shape}")
        print(f"Sample output values (result[0,0,0,:]): {result[0][0, 0, 0, :]}")

        # Verify expected results for fmod operation
        test_x_broadcast = np.broadcast_to(test_x, test_y.shape)
        expected_result = np.fmod(test_x_broadcast, test_y)
        np.testing.assert_allclose(result[0], expected_result, rtol=1e-5)
        print("Test passed: Results match expected fmod values")

    except ImportError:
        print("onnx.reference not available, skipping inference test")

if __name__ == '__main__':
    main()