#!/usr/bin/env python3

# used to generate model: onnx-tests/tests/range/range.onnx

import numpy as np
import onnx
from onnx import helper, TensorProto
from onnx.reference import ReferenceEvaluator

def main():
    node = onnx.helper.make_node(
        'Range',
        name='range',
        inputs=['start', 'end', 'step'],
        outputs=['output']
    )

    graph_def = helper.make_graph(
        nodes=[node],
        name='RangeGraph',
        inputs=[
            helper.make_tensor_value_info('start', TensorProto.INT64, []),
            helper.make_tensor_value_info('end', TensorProto.INT64, []),
            helper.make_tensor_value_info('step', TensorProto.INT64, [])
        ],
        outputs=[
            helper.make_tensor_value_info('output', TensorProto.INT64, [None])
        ],
    )

    model_def = helper.make_model(graph_def, producer_name='range')
    model_def.opset_import[0].version = 16

    onnx.save(model_def, 'range.onnx')
    print("Model saved to range.onnx")
    
    # Test with ReferenceEvaluator
    sess = ReferenceEvaluator(model_def)
    
    start = np.array(0, dtype=np.int64)
    end = np.array(5, dtype=np.int64)
    step = np.array(1, dtype=np.int64)
    
    result = sess.run(None, {'start': start, 'end': end, 'step': step})
    expected = np.arange(0, 5, 1, dtype=np.int64)
    
    print(f"Result: {result[0]}")
    print(f"Expected: {expected}")
    assert np.array_equal(result[0], expected), f"Mismatch: got {result[0]}, expected {expected}"
    print("Test passed!")

if __name__ == '__main__':
    main()
