#!/usr/bin/env python3

# used to generate model: onnx-tests/tests/constant/constant_shape.onnx

import numpy as np
import onnx
from onnx import helper, TensorProto

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
    model_def = helper.make_model(graph_def, producer_name='onnx-tests')
    model_def.opset_import[0].version = 16
    
    # Save the model
    onnx_name = "constant_shape.onnx"
    onnx.save(model_def, onnx_name)
    print("Finished exporting model to {}".format(onnx_name))
    
    # Print test expectations
    print("Input shape: [2, 4, 6]")
    print("Scalar constant: 2")
    print("Shape constant: [1, 2, 3]")
    print("Expected outputs:")
    print("  shape_add_scalar: [4, 6, 8] (shape + 2)")
    print("  shape_mul_scalar: [4, 8, 12] (shape * 2)")
    print("  shape_add_shape: [3, 6, 9] (shape + [1, 2, 3])")
    print("  shape_mul_shape: [2, 8, 18] (shape * [1, 2, 3])")

if __name__ == '__main__':
    main()