#!/usr/bin/env python3

# used to generate model: onnx-tests/tests/sum/sum.onnx

import onnx
import onnx.helper
import onnx.checker
import numpy as np

# Create input tensors
input1 = onnx.helper.make_tensor_value_info('input1', onnx.TensorProto.INT64, [3])
input2 = onnx.helper.make_tensor_value_info('input2', onnx.TensorProto.INT64, [3])
input3 = onnx.helper.make_tensor_value_info('input3', onnx.TensorProto.INT64, [3])

# Create output tensor
output = onnx.helper.make_tensor_value_info('output', onnx.TensorProto.INT64, [3])

# Create the Sum node
sum_node = onnx.helper.make_node(
    'Sum',
    inputs=['input1', 'input2', 'input3'],
    outputs=['output']
)

# Create the graph (GraphProto)
graph_def = onnx.helper.make_graph(
    nodes=[sum_node],
    name='SumGraph',
    inputs=[input1, input2, input3],
    outputs=[output]
)

# Create the model (ModelProto)
model_def = onnx.helper.make_model(graph_def, producer_name='sum-model')
onnx.checker.check_model(model_def)

# Save the ONNX model
onnx.save(model_def, 'sum_int.onnx')

print("ONNX model 'sum_int.onnx' generated successfully.")
