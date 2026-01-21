#!/usr/bin/env python3
import numpy as np
import onnx
from onnx import helper, TensorProto

# Create a simple model with Identity operation WITHOUT a constant input
# This Identity takes a graph input and passes it through

input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [3, 4])
output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [3, 4])

# Create Identity node that takes graph input (not a constant)
identity_node = helper.make_node(
    "Identity",
    inputs=["input"],
    outputs=["output"],
    name="identity_test"
)

# Create the graph
graph = helper.make_graph(
    [identity_node],
    "identity_only",
    [input_tensor],
    [output_tensor],
)

# Create the model
model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 16)])
onnx.save(model, "identity_only.onnx")
print("Created identity_only.onnx - Identity without constant input")