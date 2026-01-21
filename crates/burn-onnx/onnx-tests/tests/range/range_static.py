#!/usr/bin/env python3

# used to generate model: onnx-tests/tests/range/range_static.onnx
# This test has all parameters as constants that will be lifted

import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
from onnx.reference import ReferenceEvaluator

def main():
    # Create constant nodes for start, limit, and delta
    start_node = helper.make_node(
        'Constant',
        inputs=[],
        outputs=['start'],
        value=numpy_helper.from_array(np.array(0, dtype=np.int64))
    )
    
    limit_node = helper.make_node(
        'Constant',
        inputs=[],
        outputs=['limit'],
        value=numpy_helper.from_array(np.array(10, dtype=np.int64))
    )
    
    delta_node = helper.make_node(
        'Constant',
        inputs=[],
        outputs=['delta'],
        value=numpy_helper.from_array(np.array(2, dtype=np.int64))
    )
    
    # Create the Range node
    range_node = helper.make_node(
        'Range',
        inputs=['start', 'limit', 'delta'],
        outputs=['output']
    )

    graph_def = helper.make_graph(
        nodes=[start_node, limit_node, delta_node, range_node],
        name='RangeStaticGraph',
        inputs=[],
        outputs=[
            helper.make_tensor_value_info('output', TensorProto.INT64, [None])
        ],
    )

    model_def = helper.make_model(graph_def, producer_name='range_static')
    model_def.opset_import[0].version = 16

    # Save the model
    onnx.save(model_def, 'range_static.onnx')
    print("Model saved to range_static.onnx")
    
    # Test with ReferenceEvaluator
    sess = ReferenceEvaluator(model_def)
    result = sess.run(None, {})
    expected = np.arange(0, 10, 2, dtype=np.int64)
    
    print(f"Result shape: {result[0].shape}")
    print(f"Result values: {result[0]}")
    print(f"Expected values: {expected}")
    assert np.array_equal(result[0], expected), f"Mismatch: got {result[0]}, expected {expected}"
    print("Test passed!")

if __name__ == '__main__':
    main()