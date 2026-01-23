#!/usr/bin/env python3

import onnx
import numpy as np
from onnx import TensorProto, helper

def create_model():
    # Create a model that generates a tensor constant (non-scalar)
    # Input is a shape tensor [2, 3] which means 2x3 tensor output
    
    # Create input (shape input)
    shape_input = helper.make_tensor_value_info(
        'shape', TensorProto.INT64, [2]  # Shape tensor with 2 elements
    )
    
    # Create output - 2x3 float tensor
    output = helper.make_tensor_value_info(
        'output', TensorProto.FLOAT, [2, 3]
    )
    
    # Create ConstantOfShape node with default value (0.0 float)
    constantofshape_node = helper.make_node(
        'ConstantOfShape',
        inputs=['shape'],
        outputs=['output'],
        name='constantofshape'
    )
    
    # Create the graph
    graph = helper.make_graph(
        [constantofshape_node],
        'constant_of_shape_tensor',
        [shape_input],
        [output]
    )
    
    # Create the model
    model = helper.make_model(graph)
    model.opset_import[0].version = 16
    
    return model

if __name__ == '__main__':
    model = create_model()
    
    # Validate
    onnx.checker.check_model(model)
    
    # Save
    onnx.save(model, 'constant_of_shape_tensor.onnx')
    print("Created constant_of_shape_tensor.onnx")