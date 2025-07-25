#!/usr/bin/env python3

# used to generate model: onnx-tests/tests/add/add_shape.onnx

import numpy as np
import onnx
from onnx import helper, TensorProto

def main():
    # Create a graph that tests both Shape+Scalar and Shape+Shape operations
    # Input tensors
    input_tensor1 = helper.make_tensor_value_info('input1', TensorProto.FLOAT, [2, 3, 4])
    input_tensor2 = helper.make_tensor_value_info('input2', TensorProto.FLOAT, [5, 6, 7])
    
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
            vals=[10]
        )
    )
    
    # Add shape with scalar (Shape + Scalar)
    add_scalar_node = helper.make_node(
        'Add',
        inputs=['shape1', 'scalar'],
        outputs=['shape_plus_scalar']
    )
    
    # Add two shapes together (Shape + Shape)
    add_shapes_node = helper.make_node(
        'Add',
        inputs=['shape1', 'shape2'],
        outputs=['shape_plus_shape']
    )
    
    # Outputs - shape arrays
    output1 = helper.make_tensor_value_info('shape_plus_scalar', TensorProto.INT64, [3])
    output2 = helper.make_tensor_value_info('shape_plus_shape', TensorProto.INT64, [3])
    
    # Create the graph
    graph_def = helper.make_graph(
        [shape_node1, shape_node2, scalar_const, add_scalar_node, add_shapes_node],
        'shape_add_test',
        [input_tensor1, input_tensor2],
        [output1, output2],
    )
    
    # Create the model
    model_def = helper.make_model(graph_def, producer_name='onnx-tests')
    model_def.opset_import[0].version = 16
    
    # Save the model
    onnx_name = "add_shape.onnx"
    onnx.save(model_def, onnx_name)
    print("Finished exporting model to {}".format(onnx_name))
    
    # Print test expectations
    print("Test input1 shape: [2, 3, 4]")
    print("Test input2 shape: [5, 6, 7]")
    print("Expected shape_plus_scalar: [12, 13, 14] (shape1 + 10)")
    print("Expected shape_plus_shape: [7, 9, 11] (shape1 + shape2)")

if __name__ == '__main__':
    main()