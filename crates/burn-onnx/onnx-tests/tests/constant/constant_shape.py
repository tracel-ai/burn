#!/usr/bin/env python3

# used to generate model: onnx-tests/tests/constant/constant_shape.onnx

import numpy as np
import onnx
from onnx import helper, TensorProto
from onnx.reference import ReferenceEvaluator

# ONNX opset version to use for model generation
OPSET_VERSION = 16

def main():
    # Create a graph that tests Shape type with Constants
    # Pattern: Input tensor -> Shape op -> Binary op with Constant
    
    # Input tensor
    input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, [2, 4, 6])
    
    # Shape operation to get shape of input
    shape_node = helper.make_node(
        'Shape',
        inputs=['input'],
        outputs=['input_shape']
    )
    
    # Constant scalar for testing operations
    scalar_const = helper.make_node(
        'Constant',
        inputs=[],
        outputs=['scalar'],
        value=helper.make_tensor(
            name='scalar_tensor',
            data_type=TensorProto.INT64,
            dims=[],
            vals=[2]
        )
    )
    
    # Constant array for shape operations
    shape_const = helper.make_node(
        'Constant',
        inputs=[],
        outputs=['shape_constant'],
        value=helper.make_tensor(
            name='shape_tensor',
            data_type=TensorProto.INT64,
            dims=[3],  # 1D tensor with 3 elements
            vals=[1, 2, 3]  # Values to add/multiply with shape
        )
    )
    
    # Test multiple operations: Shape + Scalar, Shape * Scalar, Shape + Shape, Shape * Shape
    
    # Shape + Scalar
    add_scalar_node = helper.make_node(
        'Add',
        inputs=['input_shape', 'scalar'],
        outputs=['shape_add_scalar']
    )
    
    # Shape * Scalar
    mul_scalar_node = helper.make_node(
        'Mul',
        inputs=['input_shape', 'scalar'],
        outputs=['shape_mul_scalar']
    )
    
    # Shape + Shape
    add_shape_node = helper.make_node(
        'Add',
        inputs=['input_shape', 'shape_constant'],
        outputs=['shape_add_shape']
    )
    
    # Shape * Shape
    mul_shape_node = helper.make_node(
        'Mul',
        inputs=['input_shape', 'shape_constant'],
        outputs=['shape_mul_shape']
    )
    
    # Define outputs
    output1 = helper.make_tensor_value_info('shape_add_scalar', TensorProto.INT64, [3])
    output2 = helper.make_tensor_value_info('shape_mul_scalar', TensorProto.INT64, [3])
    output3 = helper.make_tensor_value_info('shape_add_shape', TensorProto.INT64, [3])
    output4 = helper.make_tensor_value_info('shape_mul_shape', TensorProto.INT64, [3])
    
    # Create the graph
    graph_def = helper.make_graph(
        [shape_node, scalar_const, shape_const, add_scalar_node, mul_scalar_node, add_shape_node, mul_shape_node],
        'constant_shape_test',
        [input_tensor],
        [output1, output2, output3, output4],
    )
    
    # Create the model
    model_def = helper.make_model(
        graph_def, 
        producer_name='onnx-tests',
        opset_imports=[helper.make_operatorsetid("", OPSET_VERSION)]
    )
    
    # Save the model
    onnx_name = "constant_shape.onnx"
    onnx.save(model_def, onnx_name)
    print("Finished exporting model to {}".format(onnx_name))
    
    # Test the model with sample data
    test_input = np.random.randn(2, 4, 6).astype(np.float32)
    
    print(f"\nTest input shape: {test_input.shape}")
    
    # Run the model using ReferenceEvaluator
    session = ReferenceEvaluator(onnx_name, verbose=0)
    outputs = session.run(None, {"input": test_input})
    
    shape_add_scalar, shape_mul_scalar, shape_add_shape, shape_mul_shape = outputs
    
    print(f"\nTest output shape_add_scalar: {repr(shape_add_scalar)}")
    print(f"Test output shape_mul_scalar: {repr(shape_mul_scalar)}")
    print(f"Test output shape_add_shape: {repr(shape_add_shape)}")
    print(f"Test output shape_mul_shape: {repr(shape_mul_shape)}")

if __name__ == '__main__':
    main()