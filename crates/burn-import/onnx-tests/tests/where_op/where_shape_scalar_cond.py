#!/usr/bin/env python3

# used to generate model: onnx-tests/tests/where_op/where_shape_scalar_cond.onnx

import numpy as np
import onnx
from onnx import helper, TensorProto
from onnx.reference import ReferenceEvaluator

# ONNX opset version to use for model generation
OPSET_VERSION = 16

def main():
    # Create a graph that tests Where operation with scalar condition and Shape x, y
    # Input tensors
    input1 = helper.make_tensor_value_info('input1', TensorProto.FLOAT, [2, 3, 4])
    input2 = helper.make_tensor_value_info('input2', TensorProto.FLOAT, [5, 6, 7])
    
    # Boolean scalar input for condition
    cond_input = helper.make_tensor_value_info('condition', TensorProto.BOOL, [])
    
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
    
    # Where operation with scalar condition and Shape x, y
    # Where(scalar_condition, shape1, shape2)
    where_node = helper.make_node(
        'Where',
        inputs=['condition', 'shape1', 'shape2'],
        outputs=['output']
    )
    
    # Output - shape array
    output = helper.make_tensor_value_info('output', TensorProto.INT64, [3])
    
    # Create the graph
    graph_def = helper.make_graph(
        [shape1, shape2, where_node],
        'where_shape_scalar_cond_test',
        [cond_input, input1, input2],
        [output],
    )
    
    # Create the model
    model_def = helper.make_model(
        graph_def,
        producer_name='onnx-tests',
        opset_imports=[helper.make_operatorsetid("", OPSET_VERSION)]
    )
    
    # Save the model
    onnx_name = "where_shape_scalar_cond.onnx"
    onnx.save(model_def, onnx_name)
    print("Finished exporting model to {}".format(onnx_name))
    
    # Test the model with sample data
    test_input1 = np.random.randn(2, 3, 4).astype(np.float32)
    test_input2 = np.random.randn(5, 6, 7).astype(np.float32)
    
    print(f"\nTest input1 shape: {test_input1.shape}")
    print(f"Test input2 shape: {test_input2.shape}")
    
    # Run the model using ReferenceEvaluator
    session = ReferenceEvaluator(onnx_name, verbose=0)
    
    # Test with condition = True
    print("\nTest with condition = True:")
    outputs_true = session.run(None, {
        "condition": np.array(True, dtype=np.bool_),
        "input1": test_input1,
        "input2": test_input2
    })
    result_true = outputs_true[0]
    print(f"Expected: shape1 [2, 3, 4]")
    print(f"Test output: {repr(result_true)}")
    
    # Test with condition = False
    print("\nTest with condition = False:")
    outputs_false = session.run(None, {
        "condition": np.array(False, dtype=np.bool_),
        "input1": test_input1,
        "input2": test_input2
    })
    result_false = outputs_false[0]
    print(f"Expected: shape2 [5, 6, 7]")
    print(f"Test output: {repr(result_false)}")

if __name__ == '__main__':
    main()