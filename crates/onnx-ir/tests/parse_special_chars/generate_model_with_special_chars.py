#!/usr/bin/env python3
"""
This script generates an ONNX model with node names containing special characters
that would cause problems for Rust identifiers, similar to what happens with tf2onnx.

The model is a simple MLP network with intentionally problematic node names.
"""

import os
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper, AttributeProto
from onnx.helper import make_node, make_graph, make_model, make_tensor_value_info

def generate_onnx_model_with_special_chars():
    # Create inputs (ValueInfoProto)
    X = make_tensor_value_info("X", TensorProto.FLOAT, [None, 3])

    # Create outputs (ValueInfoProto)
    Y = make_tensor_value_info("Y", TensorProto.FLOAT, [None, 2])

    # Create initializers
    weight1_data = np.random.randn(3, 4).astype(np.float32)
    weight1 = numpy_helper.from_array(weight1_data, name="jax2tf_w1/weight:0")

    bias1_data = np.random.randn(4).astype(np.float32)
    bias1 = numpy_helper.from_array(bias1_data, name="jax2tf_b1/bias:0")

    weight2_data = np.random.randn(4, 2).astype(np.float32)
    weight2 = numpy_helper.from_array(weight2_data, name="inception_resnet_v1/weight:0")

    bias2_data = np.random.randn(2).astype(np.float32)
    bias2 = numpy_helper.from_array(bias2_data, name="inception_resnet_v1/bias:0")

    # Create nodes with problematic names
    matmul1 = make_node(
        "MatMul",
        ["X", "jax2tf_w1/weight:0"],
        ["jax2tf_rhs_/mul/Const:0"],
        name="jax2tf_rhs_/MatMul:0"
    )

    add1 = make_node(
        "Add",
        ["jax2tf_rhs_/mul/Const:0", "jax2tf_b1/bias:0"],
        ["jax2tf_rhs_/pjit_silu_/Const_2:0"],
        name="jax2tf_rhs_/Add:0"
    )

    relu = make_node(
        "Relu",
        ["jax2tf_rhs_/pjit_silu_/Const_2:0"],
        ["inception_resnet_v1/lambda_23/mul/x:0"],
        name="jax2tf_rhs_/Relu:0"
    )

    matmul2 = make_node(
        "MatMul",
        ["inception_resnet_v1/lambda_23/mul/x:0", "inception_resnet_v1/weight:0"],
        ["inception_resnet_v1/lambda_23/mul/y:0"],
        name="inception_resnet_v1/MatMul:0"
    )

    add2 = make_node(
        "Add",
        ["inception_resnet_v1/lambda_23/mul/y:0", "inception_resnet_v1/bias:0"],
        ["Y"],
        name="inception_resnet_v1/Add:0"
    )

    # Create the graph
    graph = make_graph(
        [matmul1, add1, relu, matmul2, add2],
        "SimpleMLP",
        [X],
        [Y],
        [weight1, bias1, weight2, bias2]
    )

    # Create the model
    model = make_model(graph, producer_name="burn-onnx-test")

    # Set the IR version and opset version
    model.ir_version = 7  # Using IR version 7
    opset = model.opset_import.add()
    opset.version = 16    # Using opset version 16 (minimum required by Burn)

    return model

if __name__ == "__main__":
    model = generate_onnx_model_with_special_chars()

    # Save the model
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "special_char_nodes.onnx")
    onnx.save(model, output_path)

    print(f"Model saved to {output_path}")

    # Verify the model
    try:
        onnx.checker.check_model(model)
        print("The model is valid!")
    except Exception as e:
        print(f"The model is invalid: {e}")
