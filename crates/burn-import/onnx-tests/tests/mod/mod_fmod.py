#!/usr/bin/env python3

# used to generate model: onnx-tests/tests/mod/mod_fmod.onnx
# Tests fmod=1 (C-style fmod)

import numpy as np
import onnx
from onnx import helper, TensorProto

def main():
    # Create ONNX model with Mod operator using fmod=1 (C-style fmod)
    # Input tensors - 3D for consistency with other tests
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 1, 4])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 1, 4])

    # Output tensor
    z = helper.make_tensor_value_info('z', TensorProto.FLOAT, [1, 1, 4])

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
        'mod_fmod_model',
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
    onnx_name = "mod_fmod.onnx"
    onnx.save(model_def, onnx_name)
    onnx.checker.check_model(onnx_name)
    print(f"Finished exporting model to {onnx_name}")

    # Test with onnx.reference.ReferenceEvaluator
    try:
        from onnx.reference import ReferenceEvaluator

        # Create test data
        test_x = np.array([[[[5.3, -5.3, 7.5, -7.5]]]]).astype(np.float32)
        test_y = np.array([[[[2.0, 2.0, 3.0, 3.0]]]]).astype(np.float32)

        # Run inference with ReferenceEvaluator
        sess = ReferenceEvaluator(model_def)
        result = sess.run(None, {"x": test_x, "y": test_y})

        print(f"Test input x: {test_x}")
        print(f"Test input y: {test_y}")
        print(f"Test output (fmod): {result[0]}")

        # Verify expected results for C-style fmod operation
        expected_result = np.fmod(test_x, test_y)
        np.testing.assert_allclose(result[0], expected_result, rtol=1e-5)
        print(f"Expected values: [1.3, -1.3, 1.5, -1.5] (sign follows dividend)")
        print("Test passed: Results match expected fmod values")

    except ImportError:
        print("onnx.reference not available, skipping inference test")

if __name__ == '__main__':
    main()