#!/usr/bin/env python3

# used to generate model: onnx-tests/tests/cumsum/cumsum_reverse.onnx
# Reverse cumsum (exclusive=0, reverse=1)
# Performs cumsum in reverse direction

import numpy as np
import onnx
from onnx import TensorProto, helper
from onnx.reference import ReferenceEvaluator

# Create input tensor (1D case)
data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)

# Create input tensor info
data_tensor = helper.make_tensor_value_info("data", TensorProto.FLOAT, list(data.shape))

# Create axis as initializer (constant)
axis_tensor = helper.make_tensor("axis", TensorProto.INT64, [], [0])

# Create output tensor info
output_tensor = helper.make_tensor_value_info(
    "output", TensorProto.FLOAT, list(data.shape)
)

# Create CumSum node (exclusive=0, reverse=1)
cumsum_node = helper.make_node(
    "CumSum", inputs=["data", "axis"], outputs=["output"], exclusive=0, reverse=1
)

# Create graph and model
graph = helper.make_graph(
    [cumsum_node],
    "cumsum-reverse-model",
    [data_tensor],
    [output_tensor],
    initializer=[axis_tensor],
)

model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 16)])
onnx.checker.check_model(model)
onnx.save(model, "cumsum_reverse.onnx")

# Use ReferenceEvaluator for expected output (spec compliant)
ref = ReferenceEvaluator(model)
output = ref.run(None, {"data": data})[0]

print("=== Values for mod.rs ===")
print(f"Input data: {data.tolist()}")
print(f"Output: {output.tolist()}")
print(f"// Input: [1., 2., 3., 4., 5.]")
print(f"// Expected output (reverse): [15., 14., 12., 9., 5.]")
print("ONNX model 'cumsum_reverse.onnx' generated successfully.")
