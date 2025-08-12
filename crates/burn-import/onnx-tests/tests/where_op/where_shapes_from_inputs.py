#!/usr/bin/env python3

# used to generate model: onnx-tests/tests/where_op/where_shapes_from_inputs.onnx

import numpy as np
import onnx
from onnx import helper, TensorProto
from onnx.reference import ReferenceEvaluator

# ONNX opset version to use for model generation
OPSET_VERSION = 16

def main():
    # Create a graph that tests Where operation with Shape comparison condition
    # Input tensors
    input1 = helper.make_tensor_value_info('input1', TensorProto.FLOAT, [1, 2, 3])
    input2 = helper.make_tensor_value_info('input2', TensorProto.FLOAT, [4, 5, 6])
    input3 = helper.make_tensor_value_info('input3', TensorProto.FLOAT, [7, 8, 9])
    input4 = helper.make_tensor_value_info('input4', TensorProto.FLOAT, [1, 0, 3])  # For comparison
    
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
    
    shape3 = helper.make_node(
        'Shape',
        inputs=['input3'],
        outputs=['shape3']
    )
    
    shape4 = helper.make_node(
        'Shape',
        inputs=['input4'],
        outputs=['shape4']
    )
    
    # Equal comparison between shape1 and shape4
    # This produces a Shape with 1s and 0s based on element-wise equality
    condition = helper.make_node(
        'Equal',
        inputs=['shape1', 'shape4'],
        outputs=['condition']
    )
    
    # Where operation with all Shape types
    # Element-wise selection: Where(condition, shape2, shape3)
    where_node = helper.make_node(
        'Where',
        inputs=['condition', 'shape2', 'shape3'],
        outputs=['output']
    )
    
    # Output - shape array
    output = helper.make_tensor_value_info('output', TensorProto.INT64, [3])
    
    # Create the graph
    graph_def = helper.make_graph(
        [shape1, shape2, shape3, shape4, condition, where_node],
        'where_shapes_from_inputs_test',
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
    onnx_name = "where_shapes_from_inputs.onnx"
    onnx.save(model_def, onnx_name)
    print("Finished exporting model to {}".format(onnx_name))
    
    # Test the model with sample data
    test_input1 = np.random.randn(1, 2, 3).astype(np.float32)
    test_input2 = np.random.randn(4, 5, 6).astype(np.float32)
    test_input3 = np.random.randn(7, 8, 9).astype(np.float32)
    test_input4 = np.random.randn(1, 0, 3).astype(np.float32)  # Shape [1, 0, 3]
    
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
    
    print(f"\nCondition: shape1 == shape4")
    print(f"  shape1: {list(test_input1.shape)}")
    print(f"  shape4: {list(test_input4.shape)}")
    print(f"  Equal: [1==1, 2==0, 3==3] = [1, 0, 1]")
    print(f"Where operation selects:")
    print(f"  [0]: cond[0]=1 -> shape2[0]=4")
    print(f"  [1]: cond[1]=0 -> shape3[1]=8")
    print(f"  [2]: cond[2]=1 -> shape2[2]=6")
    print(f"Expected: [4, 8, 6]")
    print(f"Test output: {repr(result)}")

if __name__ == '__main__':
    main()