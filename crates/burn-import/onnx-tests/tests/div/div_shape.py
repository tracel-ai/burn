#!/usr/bin/env python3

# used to generate model: onnx-tests/tests/div/div_shape.onnx

import numpy as np
import onnx
from onnx import helper, TensorProto
from onnx.reference import ReferenceEvaluator

# ONNX opset version to use for model generation
OPSET_VERSION = 16

def main():
    # Create a graph that tests both Shape/Scalar and Shape/Shape operations
    # Input tensors with dimensions that are multiples
    input_tensor1 = helper.make_tensor_value_info('input1', TensorProto.FLOAT, [8, 12, 16])
    input_tensor2 = helper.make_tensor_value_info('input2', TensorProto.FLOAT, [2, 3, 4])
    
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
    
    # Divide shape by scalar (Shape / Scalar)
    div_scalar_node = helper.make_node(
        'Div',
        inputs=['shape1', 'scalar'],
        outputs=['shape_div_scalar']
    )
    
    # Divide shape by shape (Shape / Shape)
    div_shapes_node = helper.make_node(
        'Div',
        inputs=['shape1', 'shape2'],
        outputs=['shape_div_shape']
    )
    
    # Outputs - shape arrays
    output1 = helper.make_tensor_value_info('shape_div_scalar', TensorProto.INT64, [3])
    output2 = helper.make_tensor_value_info('shape_div_shape', TensorProto.INT64, [3])
    
    # Create the graph
    graph_def = helper.make_graph(
        [shape_node1, shape_node2, scalar_const, div_scalar_node, div_shapes_node],
        'shape_div_test',
        [input_tensor1, input_tensor2],
        [output1, output2],
    )
    
    # Create the model
    model_def = helper.make_model(
        graph_def, 
        producer_name='onnx-tests',
        opset_imports=[helper.make_operatorsetid("", OPSET_VERSION)]
    )
    
    # Save the model
    onnx_name = "div_shape.onnx"
    onnx.save(model_def, onnx_name)
    print("Finished exporting model to {}".format(onnx_name))
    
    # Test the model with sample data
    test_input1 = np.random.randn(8, 12, 16).astype(np.float32)
    test_input2 = np.random.randn(2, 3, 4).astype(np.float32)
    
    print(f"\nTest input1 shape: {test_input1.shape}")
    print(f"Test input2 shape: {test_input2.shape}")
    
    # Run the model using ReferenceEvaluator
    session = ReferenceEvaluator(onnx_name, verbose=0)
    outputs = session.run(None, {"input1": test_input1, "input2": test_input2})
    
    shape_div_scalar, shape_div_shape = outputs
    
    print(f"\nTest output shape_div_scalar: {repr(shape_div_scalar)}")
    print(f"Test output shape_div_shape: {repr(shape_div_shape)}")

if __name__ == '__main__':
    main()