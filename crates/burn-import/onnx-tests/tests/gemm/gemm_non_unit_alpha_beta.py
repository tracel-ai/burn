#!/usr/bin/env python3
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

def create_gemm_model(output_path="gemm_non_unit_alpha_beta.onnx"):
    """
    Create an ONNX model with a Gemm node that performs:
    Y = alpha * (A @ B) + beta * C

    Args:
        output_path (str): Path to save the ONNX model
    """
    # Define input and output shapes
    # batch_size = 1
    m, k, n = 2, 2, 2  # A: (m, k), B: (k, n), C: (m, n)

    # Define the graph inputs and outputs
    A = helper.make_tensor_value_info('A', TensorProto.FLOAT, [m, k])
    B = helper.make_tensor_value_info('B', TensorProto.FLOAT, [k, n])
    C = helper.make_tensor_value_info('C', TensorProto.FLOAT, [])
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [m, n])

    # Define Gemm node attributes
    alpha = 0.5
    beta = 0.5
    transA = 0  # 0 means no transpose
    transB = 0  # 0 means no transpose

    # Create the Gemm node
    gemm_node = helper.make_node(
        'Gemm',                # op_type
        ['A', 'B', 'C'],       # inputs
        ['Y'],                 # outputs
        name='GemmNode',       # name
        alpha=alpha,           # attributes
        beta=beta,
        transA=transA,
        transB=transB
    )

    # Create the graph
    graph = helper.make_graph(
        [gemm_node],           # nodes
        'GemmModel',           # name
        [A, B, C],             # inputs
        [Y],                   # outputs
    )

    # Create the model
    model = helper.make_model(
        graph,
        producer_name='ONNX_Generator',
        opset_imports=[helper.make_opsetid("", 16)]  # Using opset 16
    )

    # Verify the model
    onnx.checker.check_model(model)

    # Save the model
    onnx.save(model, output_path)
    print(f"Successfully created ONNX model with Gemm node at: {output_path}")

    return model

if __name__ == '__main__':
    create_gemm_model()
