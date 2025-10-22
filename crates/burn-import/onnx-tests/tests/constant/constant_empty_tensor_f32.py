#!/usr/bin/env python3

import onnx
import onnx.helper as helper
import onnx.checker as checker
import numpy as np
from onnx import TensorProto, AttributeProto

def create_model():
    """Create an ONNX model with an empty f32 tensor constant as an attribute.

    This test case reproduces a bug where ONNX constants with Float32 type
    but no data (empty float_data and empty raw_data) would cause a panic.
    """

    # Create an empty tensor as a constant attribute
    empty_tensor_attr = helper.make_tensor(
        name='empty_tensor',
        data_type=TensorProto.FLOAT,
        dims=[0],
        vals=[]
    )

    # Create a Constant node with the empty tensor
    constant_node = helper.make_node(
        'Constant',
        inputs=[],
        outputs=['empty_constant'],
        value=empty_tensor_attr
    )

    # Create an Identity node to pass through the input
    identity_node = helper.make_node(
        'Identity',
        inputs=['input'],
        outputs=['output']
    )

    # Create graph (the empty constant is created but not used in the output)
    graph_def = helper.make_graph(
        [constant_node, identity_node],
        'constant_empty_tensor_f32_model',
        [helper.make_tensor_value_info('input', TensorProto.FLOAT, [2, 3])],
        [helper.make_tensor_value_info('output', TensorProto.FLOAT, [2, 3])]
    )

    # Create model with opset version 16
    model_def = helper.make_model(
        graph_def,
        producer_name='constant_empty_tensor_f32_test',
        opset_imports=[helper.make_opsetid("", 16)]
    )

    # Check model
    checker.check_model(model_def)

    return model_def

if __name__ == '__main__':
    model = create_model()
    onnx.save(model, 'constant_empty_tensor_f32.onnx')
    print("Created constant_empty_tensor_f32.onnx")
