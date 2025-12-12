#!/usr/bin/env python3

# used to generate model: onnx-tests/tests/cumsum/cumsum_runtime_axis.onnx
# CumSum with runtime axis (axis as model input, not constant)

import numpy as np
import onnx
from onnx import TensorProto, helper
from onnx.reference import ReferenceEvaluator

# Create input tensor (2D case)
data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)

# Create input tensor info
data_tensor = helper.make_tensor_value_info("data", TensorProto.FLOAT, list(data.shape))

# Create axis as MODEL INPUT (not initializer) - this makes it a runtime value
axis_tensor = helper.make_tensor_value_info("axis", TensorProto.INT64, [])

# Create output tensor info
output_tensor = helper.make_tensor_value_info(
    "output", TensorProto.FLOAT, list(data.shape)
)

# Create CumSum node (default: exclusive=0, reverse=0)
cumsum_node = helper.make_node(
    "CumSum", inputs=["data", "axis"], outputs=["output"], exclusive=0, reverse=0
)

# Create graph and model - axis is an input, NOT an initializer
graph = helper.make_graph(
    [cumsum_node],
    "cumsum-runtime-axis-model",
    [data_tensor, axis_tensor],  # Both data and axis are inputs
    [output_tensor],
    initializer=[],  # No initializers - axis is runtime
)

model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 16)])
onnx.checker.check_model(model)
onnx.save(model, "cumsum_runtime_axis.onnx")

# Use ReferenceEvaluator for expected output (spec compliant)
# Test with axis=1
axis = np.array(1, dtype=np.int64)
ref = ReferenceEvaluator(model)
output = ref.run(None, {"data": data, "axis": axis})[0]

print("=== Values for mod.rs ===")
print(f"Input data: {data.tolist()}")
print(f"Axis: {axis.item()}")
print(f"Output: {output.tolist()}")
print(f"// Input: [[1., 2., 3.], [4., 5., 6.]]")
print(f"// Axis: 1 (runtime)")
print(f"// Expected output (2D, axis=1): [[1., 3., 6.], [4., 9., 15.]]")
print("ONNX model 'cumsum_runtime_axis.onnx' generated successfully.")
