#!/usr/bin/env python3

import onnx
import numpy as np
from onnx import TensorProto, helper

def create_model():
    # Create a model that generates a scalar constant with custom value
    # Shape is provided as an initializer (compile-time constant)

    # Create shape initializer - empty array means scalar output
    shape_initializer = helper.make_tensor(
        name='shape',
        data_type=TensorProto.INT64,
        dims=[0],  # 1D tensor with 0 elements (empty array)
        vals=[]
    )

    # Create output - scalar int64
    output = helper.make_tensor_value_info(
        'output', TensorProto.INT64, []  # Empty shape means scalar
    )

    # Create a tensor for the custom value (42 as int64)
    value_tensor = helper.make_tensor(
        name='value',
        data_type=TensorProto.INT64,
        dims=[],  # Scalar value
        vals=[42]
    )

    # Create ConstantOfShape node with custom value
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
        'constant_of_shape_scalar_custom_value',
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
    onnx.save(model, 'constant_of_shape_scalar_custom_value.onnx')
    print("Created constant_of_shape_scalar_custom_value.onnx")