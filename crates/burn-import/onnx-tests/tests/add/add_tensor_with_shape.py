#!/usr/bin/env python3

# used to generate model: onnx-tests/tests/add/add_tensor_with_shape.onnx

import numpy as np
import onnx
from onnx import helper, TensorProto
from onnx.reference import ReferenceEvaluator

# ONNX opset version to use for model generation
OPSET_VERSION = 16

def main():
    # Create a graph that tests Tensor+Shape operations
    # This ensures the tensor stays as a tensor by using it in other operations
    
    # Input tensors
    input_tensor = helper.make_tensor_value_info('input_tensor', TensorProto.FLOAT, [2, 3, 4])
    input_1d = helper.make_tensor_value_info('input_1d', TensorProto.INT64, [3])
    
    # Shape node - extract shape of input
    shape_node = helper.make_node(
        'Shape',
        inputs=['input_tensor'],
        outputs=['shape']
    )
    
    # Use the 1D tensor in another operation first to ensure it stays as tensor
    # ArgMax to get a scalar/1D result from the tensor
    argmax_node = helper.make_node(
        'ArgMax',
        inputs=['input_1d'],
        outputs=['argmax_result'],
        axis=0,
        keepdims=False
    )
    
    # Cast argmax result to int64 tensor
    cast_node = helper.make_node(
        'Cast',
        inputs=['argmax_result'],
        outputs=['argmax_int'],
        to=TensorProto.INT64
    )
    
    # Range operation using argmax result 
    range_const_start = helper.make_node(
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
    
    range_const_delta = helper.make_node(
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
    
    # Create a range tensor
    range_node = helper.make_node(
        'Range',
        inputs=['zero', 'input_1d', 'one'],
        outputs=['range_tensor']
    )
    
    # Gather from shape using constant index to get a scalar shape element
    one_const = helper.make_node(
        'Constant',
        inputs=[],
        outputs=['const_one_idx'],
        value=helper.make_tensor(
            name='const_one_idx_val',
            data_type=TensorProto.INT64,
            dims=[],
            vals=[1]
        )
    )
    
    gather_node = helper.make_node(
        'Gather',
        inputs=['shape', 'const_one_idx'],
        outputs=['shape_elem'],
        axis=0
    )
    
    # Create range using the gathered shape element
    range_node2 = helper.make_node(
        'Range',
        inputs=['zero', 'shape_elem', 'one'],
        outputs=['range_tensor']
    )
    
    # Multiply the original input_1d tensor with a constant to ensure it stays as tensor
    const_multiplier = helper.make_node(
        'Constant',
        inputs=[],
        outputs=['multiplier'],
        value=helper.make_tensor(
            name='const_mult',
            data_type=TensorProto.INT64,
            dims=[],
            vals=[2]
        )
    )
    
    mul_node = helper.make_node(
        'Mul',
        inputs=['input_1d', 'multiplier'],
        outputs=['mul_result']
    )
    
    # Now add the multiplication result (tensor) with the shape
    add_node = helper.make_node(
        'Add',
        inputs=['mul_result', 'shape'],
        outputs=['tensor_plus_shape']
    )
    
    # Output
    output = helper.make_tensor_value_info('tensor_plus_shape', TensorProto.INT64, [3])
    
    # Create the graph
    graph_def = helper.make_graph(
        [shape_node, argmax_node, cast_node, range_const_start, range_const_delta, 
         range_node, one_const, gather_node, range_node2, const_multiplier, mul_node, add_node],
        'tensor_with_shape_add_test',
        [input_tensor, input_1d],
        [output],
    )
    
    # Create the model
    model_def = helper.make_model(
        graph_def, 
        producer_name='onnx-tests',
        opset_imports=[helper.make_operatorsetid("", OPSET_VERSION)]
    )
    
    # Save the model
    onnx_name = "add_tensor_with_shape.onnx"
    onnx.save(model_def, onnx_name)
    print("Finished exporting model to {}".format(onnx_name))
    
    # Test the model with sample data
    test_input = np.random.randn(2, 3, 4).astype(np.float32)
    test_1d = np.array([2, 3, 4], dtype=np.int64)
    
    print(f"\nTest input shape: {test_input.shape}")
    print(f"Test 1d tensor: {test_1d}")
    
    # Run the model using ReferenceEvaluator
    session = ReferenceEvaluator(onnx_name, verbose=0)
    outputs = session.run(None, {"input_tensor": test_input, "input_1d": test_1d})
    
    result = outputs[0]
    
    print(f"\nTest output tensor_plus_shape: {repr(result)}")

if __name__ == '__main__':
    main()