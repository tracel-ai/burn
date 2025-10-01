#!/usr/bin/env python3

# used to generate model: onnx-tests/tests/range/range_runtime.onnx  
# This test has all parameters as runtime inputs (no constant lifting)

import numpy as np
import onnx
from onnx import helper, TensorProto
from onnx.reference import ReferenceEvaluator

def main():
    # Create the Range node with all runtime inputs
    range_node = helper.make_node(
        'Range',
        inputs=['start', 'limit', 'delta'],
        outputs=['output']
    )

    graph_def = helper.make_graph(
        nodes=[range_node],
        name='RangeRuntimeGraph',
        inputs=[
            helper.make_tensor_value_info('start', TensorProto.INT64, []),
            helper.make_tensor_value_info('limit', TensorProto.INT64, []),
            helper.make_tensor_value_info('delta', TensorProto.INT64, [])
        ],
        outputs=[
            helper.make_tensor_value_info('output', TensorProto.INT64, [None])
        ],
    )

    model_def = helper.make_model(graph_def, producer_name='range_runtime')
    model_def.opset_import[0].version = 16

    # Save the model
    onnx.save(model_def, 'range_runtime.onnx')
    print("Model saved to range_runtime.onnx")
    
    # Test with ReferenceEvaluator
    sess = ReferenceEvaluator(model_def)
    
    # Test case 1: 0 to 10 by 2
    start = np.array(0, dtype=np.int64)
    limit = np.array(10, dtype=np.int64)
    delta = np.array(2, dtype=np.int64)
    result = sess.run(None, {'start': start, 'limit': limit, 'delta': delta})
    expected = np.arange(0, 10, 2, dtype=np.int64)
    print(f"\nTest 1 - range(0, 10, 2):")
    print(f"Result: {result[0]}")
    print(f"Expected: {expected}")
    assert np.array_equal(result[0], expected), f"Mismatch: got {result[0]}, expected {expected}"
    
    # Test case 2: 5 to 20 by 3
    start = np.array(5, dtype=np.int64)
    limit = np.array(20, dtype=np.int64)
    delta = np.array(3, dtype=np.int64)
    result = sess.run(None, {'start': start, 'limit': limit, 'delta': delta})
    expected = np.arange(5, 20, 3, dtype=np.int64)
    print(f"\nTest 2 - range(5, 20, 3):")
    print(f"Result: {result[0]}")
    print(f"Expected: {expected}")
    assert np.array_equal(result[0], expected), f"Mismatch: got {result[0]}, expected {expected}"
    
    # Test case 3: Single element range
    start = np.array(10, dtype=np.int64)
    limit = np.array(11, dtype=np.int64)
    delta = np.array(1, dtype=np.int64)
    result = sess.run(None, {'start': start, 'limit': limit, 'delta': delta})
    expected = np.arange(10, 11, 1, dtype=np.int64)
    print(f"\nTest 3 - range(10, 11, 1):")
    print(f"Result: {result[0]}")
    print(f"Expected: {expected}")
    assert np.array_equal(result[0], expected), f"Mismatch: got {result[0]}, expected {expected}"
    
    print("\nAll tests passed!")

if __name__ == '__main__':
    main()