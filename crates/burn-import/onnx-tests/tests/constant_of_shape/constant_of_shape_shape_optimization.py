#!/usr/bin/env python3

import onnx
import numpy as np
from onnx import TensorProto, helper

def create_model():
    # Create a model that tests the Shape(1) optimization
    # Shape is provided as an initializer with value [3]
    # Value is Int64, output should be Shape(1) optimized to [i64; 1]

    # Create shape initializer - [3] means create 1D array with 3 elements
    shape_initializer = helper.make_tensor(
        name='shape',
        data_type=TensorProto.INT64,
        dims=[1],  # 1D tensor with 1 element
        vals=[3]   # The shape will be [3]
    )

    # Create output - 1D int64 tensor with 3 elements
    output = helper.make_tensor_value_info(
        'output', TensorProto.INT64, [3]
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

    # Create the graph with initializer instead of input
    graph = helper.make_graph(
        [constantofshape_node],
        'constant_of_shape_shape_optimization',
        [],  # No runtime inputs
        [output],
        initializer=[shape_initializer]
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