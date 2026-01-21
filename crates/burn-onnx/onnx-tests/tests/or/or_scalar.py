#!/usr/bin/env python3
"""Generate or_scalar.onnx using only the onnx library (no PyTorch required)"""

import onnx
from onnx import helper, TensorProto

# Create two constant nodes with scalar boolean values
constant1_node = helper.make_node(
    'Constant',
    inputs=[],
    outputs=['constant1'],
    value=helper.make_tensor(
        name='const1',
        data_type=TensorProto.BOOL,
        dims=[],  # Scalar
        vals=[True]
    )
)

constant2_node = helper.make_node(
    'Constant',
    inputs=[],
    outputs=['constant2'],
    value=helper.make_tensor(
        name='const2',
        data_type=TensorProto.BOOL,
        dims=[],  # Scalar
        vals=[False]
    )
)

# Create Or node for scalar inputs
or1_node = helper.make_node(
    'Or',
    inputs=['input1', 'constant1'],
    outputs=['or1_output']
)

or2_node = helper.make_node(
    'Or',
    inputs=['input2', 'constant2'],
    outputs=['or2_output']
)

or3_node = helper.make_node(
    'Or',
    inputs=['or1_output', 'or2_output'],
    outputs=['output']
)

# Create the graph
graph_def = helper.make_graph(
    [constant1_node, constant2_node, or1_node, or2_node, or3_node],
    'or_scalar_model',
    [
        helper.make_tensor_value_info('input1', TensorProto.BOOL, []),  # Scalar bool input
        helper.make_tensor_value_info('input2', TensorProto.BOOL, [])   # Scalar bool input
    ],
    [helper.make_tensor_value_info('output', TensorProto.BOOL, [])]     # Scalar bool output
)

# Create the model
model_def = helper.make_model(
    graph_def,
    producer_name='or_scalar_test',
    opset_imports=[helper.make_opsetid("", 16)]
)

# Save the model
onnx.save(model_def, 'or_scalar.onnx')
print("Generated or_scalar.onnx")