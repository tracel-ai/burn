#!/usr/bin/env python3

# used to generate model: onnx-tests/tests/mul/mul_shape.onnx

import numpy as np
import onnx
from onnx import helper, TensorProto

def main():
    # Create a graph that tests both Shape*Scalar and Shape*Shape operations
    # Input tensors
    input_tensor1 = helper.make_tensor_value_info('input1', TensorProto.FLOAT, [2, 3, 4])
    input_tensor2 = helper.make_tensor_value_info('input2', TensorProto.FLOAT, [1, 2, 3])
    
    # Shape nodes - extract shapes of inputs
    shape_node1 = helper.make_node(
        'Shape',
        inputs=['input1'],
        outputs=['shape1']
    )
    
    shape_node2 = helper.make_node(
        'Shape',
        inputs=['input2'],
        outputs=['shape2']
    )
    
    # Constant scalar value
    scalar_const = helper.make_node(
        'Constant',
        inputs=[],
        outputs=['scalar'],
        value=helper.make_tensor(
            name='const_tensor',
            data_type=TensorProto.INT64,
            dims=[],
            vals=[2]
        )
    )
    
    # Multiply shape by scalar (Shape * Scalar)
    mul_scalar_node = helper.make_node(
        'Mul',
        inputs=['shape1', 'scalar'],
        outputs=['shape_times_scalar']
    )
    
    # Multiply shape by shape (Shape * Shape)
    mul_shapes_node = helper.make_node(
        'Mul',
        inputs=['shape1', 'shape2'],
        outputs=['shape_times_shape']
    )
    
    # Outputs - shape arrays
    output1 = helper.make_tensor_value_info('shape_times_scalar', TensorProto.INT64, [3])
    output2 = helper.make_tensor_value_info('shape_times_shape', TensorProto.INT64, [3])
    
    # Create the graph
    graph_def = helper.make_graph(
        [shape_node1, shape_node2, scalar_const, mul_scalar_node, mul_shapes_node],
        'shape_mul_test',
        [input_tensor1, input_tensor2],
        [output1, output2],
    )
    
    # Create the model
    model_def = helper.make_model(graph_def, producer_name='onnx-tests')
    model_def.opset_import[0].version = 16
    
    # Save the model
    onnx_name = "mul_shape.onnx"
    onnx.save(model_def, onnx_name)
    print("Finished exporting model to {}".format(onnx_name))
    
    # Print test expectations
    print("Test input1 shape: [2, 3, 4]")
    print("Test input2 shape: [1, 2, 3]")
    print("Expected shape_times_scalar: [4, 6, 8] (shape1 * 2)")
    print("Expected shape_times_shape: [2, 6, 12] (shape1 * shape2)")

if __name__ == '__main__':
    main()