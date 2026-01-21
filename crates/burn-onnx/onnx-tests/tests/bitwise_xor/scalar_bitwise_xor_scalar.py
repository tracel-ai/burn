#!/usr/bin/env python3

import onnx
import onnx.helper as helper
import onnx.checker as checker
import onnx.numpy_helper
import numpy as np

def build_model():
    # Scalar inputs
    input1 = helper.make_tensor_value_info("input1", onnx.TensorProto.INT32, [])
    input2 = helper.make_tensor_value_info("input2", onnx.TensorProto.INT32, [])
    output = helper.make_tensor_value_info("output", onnx.TensorProto.INT32, [])

    # Create bitwise XOR node
    xor_node = helper.make_node(
        "BitwiseXor",
        inputs=["input1", "input2"],
        outputs=["output"]
    )

    # Create the graph
    graph_def = helper.make_graph(
        [xor_node],
        "scalar_bitwise_xor_scalar",
        [input1, input2],
        [output],
    )

    # Create the model
    model_def = helper.make_model(graph_def, producer_name="scalar_bitwise_xor_scalar")
    checker.check_model(model_def)

    return model_def

if __name__ == "__main__":
    model = build_model()
    onnx.save(model, "scalar_bitwise_xor_scalar.onnx")