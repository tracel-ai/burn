#!/usr/bin/env python3

# used to generate model: onnx-tests/tests/add/add_argmax_with_shape.onnx
# This replicates the exact pattern from the CLIP model that was failing

import numpy as np
import onnx
from onnx import helper, TensorProto
from onnx.reference import ReferenceEvaluator

# ONNX opset version to use for model generation
OPSET_VERSION = 16

def main():
    # Create a graph that tests the exact pattern from CLIP model:
    # ArgMax output (tensor) + Mul of Range and Gather (shape)
    
    # Input tensor
    input_tensor = helper.make_tensor_value_info('input_tensor', TensorProto.FLOAT, [3, 4])
    
    # ArgMax to get indices (produces a tensor)
    argmax_node = helper.make_node(
        'ArgMax',
        inputs=['input_tensor'],
        outputs=['argmax_result'],
        axis=1,
        keepdims=False
    )
    
    # Shape node - extract shape of input
    shape_node = helper.make_node(
        'Shape',
        inputs=['input_tensor'],
        outputs=['shape']
    )
    
    # Gather second dimension from shape
    const_one = helper.make_node(
        'Constant',
        inputs=[],
        outputs=['one'],
        value=helper.make_tensor(
            name='const_one',
            data_type=TensorProto.INT64,
            dims=[],
            vals=[1]
        )
    )
    
    gather_node = helper.make_node(
        'Gather',
        inputs=['shape', 'one'],
        outputs=['shape_dim'],
        axis=0
    )
    
    # Create a range
    const_zero = helper.make_node(
        'Constant',
        inputs=[],
        outputs=['zero'],
        value=helper.make_tensor(
            name='const_zero',
            data_type=TensorProto.INT64,
            dims=[],
            vals=[0]
        )
    )
    
    const_three = helper.make_node(
        'Constant',
        inputs=[],
        outputs=['three'],
        value=helper.make_tensor(
            name='const_three',
            data_type=TensorProto.INT64,
            dims=[],
            vals=[3]
        )
    )
    
    range_node = helper.make_node(
        'Range',
        inputs=['zero', 'three', 'one'],
        outputs=['range_result']
    )
    
    # Multiply range with shape dimension (produces shape type)
    mul_node = helper.make_node(
        'Mul',
        inputs=['range_result', 'shape_dim'],
        outputs=['mul_result']
    )
    
    # Add argmax result (tensor) with multiplication result (shape)
    # This is the operation that was failing
    add_node = helper.make_node(
        'Add',
        inputs=['argmax_result', 'mul_result'],
        outputs=['final_result']
    )
    
    # Output
    output = helper.make_tensor_value_info('final_result', TensorProto.INT64, [3])
    
    # Create the graph
    graph_def = helper.make_graph(
        [argmax_node, shape_node, const_one, gather_node, const_zero, const_three,
         range_node, mul_node, add_node],
        'argmax_shape_add_test',
        [input_tensor],
        [output],
    )
    
    # Create the model
    model_def = helper.make_model(
        graph_def, 
        producer_name='onnx-tests',
        opset_imports=[helper.make_operatorsetid("", OPSET_VERSION)]
    )
    
    # Save the model
    onnx_name = "add_argmax_with_shape.onnx"
    onnx.save(model_def, onnx_name)
    print("Finished exporting model to {}".format(onnx_name))
    
    # Test the model with sample data
    test_input = np.array([[1.0, 3.0, 2.0, 4.0],
                           [5.0, 2.0, 6.0, 3.0],
                           [2.0, 4.0, 1.0, 5.0]], dtype=np.float32)
    
    print(f"\nTest input:\n{test_input}")
    
    # Run the model using ReferenceEvaluator
    session = ReferenceEvaluator(onnx_name, verbose=0)
    outputs = session.run(None, {"input_tensor": test_input})
    
    result = outputs[0]
    
    print(f"\nTest output final_result: {repr(result)}")
    
    # Verify: argmax should be [3, 2, 3], range*shape_dim should be [0, 4, 8]
    # Result should be [3, 6, 11]
    expected = np.array([3, 6, 11], dtype=np.int64)
    assert np.array_equal(result, expected), f"Expected {expected}, got {result}"
    print(f"Test passed! Result matches expected: {expected}")

if __name__ == '__main__':
    main()