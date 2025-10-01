#!/usr/bin/env python3
"""Generate and_scalar.onnx using only the onnx library (no PyTorch required)"""

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

# Create And node for scalar inputs
and1_node = helper.make_node(
    'And',
    inputs=['input1', 'constant1'],
    outputs=['and1_output']
)

and2_node = helper.make_node(
    'And',
    inputs=['input2', 'constant2'],
    outputs=['and2_output']
)

and3_node = helper.make_node(
    'And',
    inputs=['and1_output', 'and2_output'],
    outputs=['output']
)

# Create the graph
graph_def = helper.make_graph(
    [constant1_node, constant2_node, and1_node, and2_node, and3_node],
    'and_scalar_model',
    [
        helper.make_tensor_value_info('input1', TensorProto.BOOL, []),  # Scalar bool input
        helper.make_tensor_value_info('input2', TensorProto.BOOL, [])   # Scalar bool input
    ],
    [helper.make_tensor_value_info('output', TensorProto.BOOL, [])]     # Scalar bool output
)

# Create the model
model_def = helper.make_model(
    graph_def,
    producer_name='and_scalar_test',
    opset_imports=[helper.make_opsetid("", 16)]
)

# Save the model
onnx.save(model_def, 'and_scalar.onnx')
print("Generated and_scalar.onnx")