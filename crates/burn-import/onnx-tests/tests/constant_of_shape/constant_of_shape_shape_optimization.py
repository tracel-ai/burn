#!/usr/bin/env python3

import onnx
import numpy as np
from onnx import TensorProto, helper

def create_model():
    # Create a model that tests the Shape(1) optimization
    # Input is Shape(1), value is Int64, output should be Shape(1)
    
    # Create input (shape input)
    shape_input = helper.make_tensor_value_info(
        'shape', TensorProto.INT64, [1]  # Shape tensor with 1 element
    )
    
    # Create output - 1D int64 tensor
    output = helper.make_tensor_value_info(
        'output', TensorProto.INT64, [1]
    )
    
    # Create a tensor for the custom value (5 as int64)
    value_tensor = helper.make_tensor(
        name='value',
        data_type=TensorProto.INT64,
        dims=[],  # Scalar value
        vals=[5]
    )
    
    # Create ConstantOfShape node with Int64 value
    constantofshape_node = helper.make_node(
        'ConstantOfShape',
        inputs=['shape'],
        outputs=['output'],
        name='constantofshape',
        value=value_tensor
    )
    
    # Create the graph
    graph = helper.make_graph(
        [constantofshape_node],
        'constant_of_shape_shape_optimization',
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
    onnx.save(model, 'constant_of_shape_shape_optimization.onnx')
    print("Created constant_of_shape_shape_optimization.onnx")