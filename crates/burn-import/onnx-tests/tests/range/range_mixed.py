#!/usr/bin/env python3

# used to generate model: onnx-tests/tests/range/range_mixed.onnx
# This test has start as runtime input, limit as constant (lifted), and delta as constant (lifted)

import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
from onnx.reference import ReferenceEvaluator

def main():
    # Create constant nodes for limit and delta (these will be lifted)
    limit_node = helper.make_node(
        'Constant',
        inputs=[],
        outputs=['limit'],
        value=numpy_helper.from_array(np.array(15, dtype=np.int64))
    )
    
    delta_node = helper.make_node(
        'Constant',
        inputs=[],
        outputs=['delta'],
        value=numpy_helper.from_array(np.array(3, dtype=np.int64))
    )
    
    # Create the Range node (start comes from input)
    range_node = helper.make_node(
        'Range',
        inputs=['start', 'limit', 'delta'],
        outputs=['output']
    )

    graph_def = helper.make_graph(
        nodes=[limit_node, delta_node, range_node],
        name='RangeMixedGraph',
        inputs=[
            helper.make_tensor_value_info('start', TensorProto.INT64, [])
        ],
        outputs=[
            helper.make_tensor_value_info('output', TensorProto.INT64, [None])
        ],
    )

    model_def = helper.make_model(graph_def, producer_name='range_mixed')
    model_def.opset_import[0].version = 16

    # Save the model
    onnx.save(model_def, 'range_mixed.onnx')
    print("Model saved to range_mixed.onnx")
    
    # Test with ReferenceEvaluator
    sess = ReferenceEvaluator(model_def)
    
    # Test case 1: start = 0
    start_val = np.array(0, dtype=np.int64)
    result = sess.run(None, {'start': start_val})
    expected = np.arange(0, 15, 3, dtype=np.int64)
    print(f"\nTest 1 - start=0:")
    print(f"Result: {result[0]}")
    print(f"Expected: {expected}")
    assert np.array_equal(result[0], expected), f"Mismatch: got {result[0]}, expected {expected}"
    
    # Test case 2: start = 3
    start_val = np.array(3, dtype=np.int64)
    result = sess.run(None, {'start': start_val})
    expected = np.arange(3, 15, 3, dtype=np.int64)
    print(f"\nTest 2 - start=3:")
    print(f"Result: {result[0]}")
    print(f"Expected: {expected}")
    assert np.array_equal(result[0], expected), f"Mismatch: got {result[0]}, expected {expected}"
    
    print("\nAll tests passed!")

if __name__ == '__main__':
    main()