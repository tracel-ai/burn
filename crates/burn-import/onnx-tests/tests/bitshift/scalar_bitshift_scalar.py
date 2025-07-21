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

    # Create bitshift left node
    bitshift_node = helper.make_node(
        "BitShift",
        inputs=["input1", "input2"],
        outputs=["output"],
        direction="LEFT"
    )

    # Create the graph
    graph_def = helper.make_graph(
        [bitshift_node],
        "scalar_bitshift_scalar",
        [input1, input2],
        [output],
    )

    # Create the model
    model_def = helper.make_model(graph_def, producer_name="scalar_bitshift_scalar")
    checker.check_model(model_def)

    return model_def

if __name__ == "__main__":
    model = build_model()
    
    # Save both left and right models
    onnx.save(model, "scalar_bitshift_left_scalar.onnx")
    
    # Create right shift model
    model_right = build_model()
    # Update direction to RIGHT
    for node in model_right.graph.node:
        if node.op_type == "BitShift":
            for attr in node.attribute:
                if attr.name == "direction":
                    attr.s = b"RIGHT"
    
    onnx.save(model_right, "scalar_bitshift_right_scalar.onnx")