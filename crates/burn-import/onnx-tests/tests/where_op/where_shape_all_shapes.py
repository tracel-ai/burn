#!/usr/bin/env python3

# used to generate model: onnx-tests/tests/where_op/where_shape_all_shapes.onnx

import numpy as np
import onnx
from onnx import helper, TensorProto
from onnx.reference import ReferenceEvaluator

# ONNX opset version to use for model generation
OPSET_VERSION = 16

def main():
    # Create a graph that tests Where operation with all Shape type inputs
    # Input tensors - we'll extract their shapes
    input1 = helper.make_tensor_value_info('input1', TensorProto.FLOAT, [2, 3, 4])
    input2 = helper.make_tensor_value_info('input2', TensorProto.FLOAT, [5, 6, 7])
    input3 = helper.make_tensor_value_info('input3', TensorProto.FLOAT, [10, 20, 30])
    input4 = helper.make_tensor_value_info('input4', TensorProto.FLOAT, [100, 200, 300])
    
    # Shape nodes - extract shapes of inputs
    shape1 = helper.make_node(
        'Shape',
        inputs=['input1'],
        outputs=['shape1']
    )
    
    shape2 = helper.make_node(
        'Shape',
        inputs=['input2'],
        outputs=['shape2']
    )
    
    shape_x = helper.make_node(
        'Shape',
        inputs=['input3'],
        outputs=['shape_x']
    )
    
    shape_y = helper.make_node(
        'Shape',
        inputs=['input4'],
        outputs=['shape_y']
    )
    
    # Equal comparison between shapes to create condition
    # This produces a Shape with 1s and 0s (1 for true, 0 for false)
    shape_equal = helper.make_node(
        'Equal',
        inputs=['shape1', 'shape2'],
        outputs=['condition_shape']
    )
    
    # Where operation with all Shape types
    # Where(condition_shape, shape_x, shape_y)
    where_node = helper.make_node(
        'Where',
        inputs=['condition_shape', 'shape_x', 'shape_y'],
        outputs=['output']
    )
    
    # Output - shape array
    output = helper.make_tensor_value_info('output', TensorProto.INT64, [3])
    
    # Create the graph
    graph_def = helper.make_graph(
        [shape1, shape2, shape_x, shape_y, shape_equal, where_node],
        'where_shape_all_shapes_test',
        [input1, input2, input3, input4],
        [output],
    )
    
    # Create the model
    model_def = helper.make_model(
        graph_def,
        producer_name='onnx-tests',
        opset_imports=[helper.make_operatorsetid("", OPSET_VERSION)]
    )
    
    # Save the model
    onnx_name = "where_shape_all_shapes.onnx"
    onnx.save(model_def, onnx_name)
    print("Finished exporting model to {}".format(onnx_name))
    
    # Test the model with sample data
    test_input1 = np.random.randn(2, 3, 4).astype(np.float32)
    test_input2 = np.random.randn(5, 6, 7).astype(np.float32)
    test_input3 = np.random.randn(10, 20, 30).astype(np.float32)
    test_input4 = np.random.randn(100, 200, 300).astype(np.float32)
    
    print(f"\nTest input1 shape: {test_input1.shape}")
    print(f"Test input2 shape: {test_input2.shape}")
    print(f"Test input3 shape: {test_input3.shape}")
    print(f"Test input4 shape: {test_input4.shape}")
    
    # Run the model using ReferenceEvaluator
    session = ReferenceEvaluator(onnx_name, verbose=0)
    outputs = session.run(None, {
        "input1": test_input1,
        "input2": test_input2,
        "input3": test_input3,
        "input4": test_input4
    })
    
    result = outputs[0]
    
    print(f"\nCondition (shape1 == shape2): shape1={list(test_input1.shape)} == shape2={list(test_input2.shape)}")
    print(f"Expected: All false -> select shape_y [100, 200, 300]")
    print(f"Test output: {repr(result)}")

if __name__ == '__main__':
    main()