#!/usr/bin/env python3

# used to generate model: onnx-tests/tests/shape/shape_chain.onnx
# This tests multiple Shape operations chained together

import onnx
from onnx import helper, TensorProto
from onnx.reference import ReferenceEvaluator
import numpy as np


def main():
    """
    Create an ONNX model with multiple chained Shape operations.
    This tests that Shape types are properly handled as inputs.
    
    Graph structure:
    input_tensor -> Shape -> shape1 (full shape)
    shape1 -> Shape -> rank_shape (shape of shape = rank)
    input_tensor -> Shape(start=0, end=2) -> partial_shape
    partial_shape -> Shape -> partial_rank_shape
    """
    
    # Create input tensor placeholder
    input_tensor = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [3, 4, 5, 6]
    )
    
    # First Shape: Get full shape of input
    shape1_node = helper.make_node(
        "Shape",
        inputs=["input"],
        outputs=["shape1"],
        name="shape1_node"
    )
    
    # Second Shape: Get shape of shape1 (returns rank)
    rank_shape_node = helper.make_node(
        "Shape",
        inputs=["shape1"],
        outputs=["rank_shape"],
        name="rank_shape_node"
    )
    
    # Third Shape: Get partial shape (first 2 dims)
    partial_shape_node = helper.make_node(
        "Shape",
        inputs=["input"],
        outputs=["partial_shape"],
        name="partial_shape_node",
        start=0,
        end=2
    )
    
    # Fourth Shape: Get shape of partial shape
    partial_rank_shape_node = helper.make_node(
        "Shape",
        inputs=["partial_shape"],
        outputs=["partial_rank_shape"],
        name="partial_rank_shape_node"
    )
    
    # Create outputs
    shape1_output = helper.make_tensor_value_info(
        "shape1", TensorProto.INT64, [4]
    )
    rank_shape_output = helper.make_tensor_value_info(
        "rank_shape", TensorProto.INT64, [1]
    )
    partial_shape_output = helper.make_tensor_value_info(
        "partial_shape", TensorProto.INT64, [2]
    )
    partial_rank_shape_output = helper.make_tensor_value_info(
        "partial_rank_shape", TensorProto.INT64, [1]
    )
    
    # Create the graph
    graph = helper.make_graph(
        [shape1_node, rank_shape_node, partial_shape_node, partial_rank_shape_node],
        "shape_chain_model",
        [input_tensor],
        [shape1_output, rank_shape_output, partial_shape_output, partial_rank_shape_output],
    )
    
    # Create the model
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 16)])
    model.ir_version = 8
    
    # Save the model
    file_name = "shape_chain.onnx"
    onnx.save(model, file_name)
    print(f"Finished exporting model to {file_name}")
    
    # Compute expected outputs using ReferenceEvaluator
    print("\nComputing expected outputs using ReferenceEvaluator:")
    
    # Create test input
    test_input_shape = (3, 4, 5, 6)
    test_input = np.random.randn(*test_input_shape).astype(np.float32)
    
    # Use ReferenceEvaluator to compute outputs
    sess = ReferenceEvaluator(model)
    outputs = sess.run(None, {"input": test_input})
    
    shape1_output = outputs[0]
    rank_shape_output = outputs[1]
    partial_shape_output = outputs[2]
    partial_rank_shape_output = outputs[3]
    
    print(f"Test input shape: {test_input.shape}")
    print(f"shape1 output: {shape1_output}")
    print(f"rank_shape output: {rank_shape_output}")
    print(f"partial_shape output: {partial_shape_output}")
    print(f"partial_rank_shape output: {partial_rank_shape_output}")
    
    # Save test data for use in Rust tests
    print(f"\nFor Rust tests:")
    print(f"  Input tensor shape: {list(test_input_shape)}")
    print(f"  shape1 should return: {shape1_output.tolist()}")
    print(f"  rank_shape should return: {rank_shape_output.tolist()}")
    print(f"  partial_shape should return: {partial_shape_output.tolist()}")
    print(f"  partial_rank_shape should return: {partial_rank_shape_output.tolist()}")


if __name__ == "__main__":
    main()