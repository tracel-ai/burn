#!/usr/bin/env python3

# used to generate model: onnx-tests/tests/shape/shape_of_shape.onnx
# This tests the Shape operation when the input is already a Shape (not a Tensor)

import onnx
from onnx import helper, TensorProto, numpy_helper
from onnx.reference import ReferenceEvaluator
import numpy as np


def main():
    """
    Create an ONNX model that applies Shape operation to a Shape output.
    This tests the fix in unary.rs for handling Shape inputs to Shape operation.
    
    Graph structure:
    input_tensor -> Shape -> shape1
    shape1 -> Shape -> shape2 (should return the rank of shape1 as [N])
    """
    
    # Create input tensor placeholder
    input_tensor = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [2, 3, 4, 5]
    )
    
    # First Shape node: Get shape of input tensor
    shape1_node = helper.make_node(
        "Shape",
        inputs=["input"],
        outputs=["shape1"],
        name="shape1_node"
    )
    
    # Second Shape node: Get shape of the shape tensor (should return [4] since rank is 4)
    shape2_node = helper.make_node(
        "Shape",
        inputs=["shape1"],
        outputs=["shape2"],
        name="shape2_node"
    )
    
    # Create outputs
    shape1_output = helper.make_tensor_value_info(
        "shape1", TensorProto.INT64, [4]  # Shape of a 4D tensor
    )
    shape2_output = helper.make_tensor_value_info(
        "shape2", TensorProto.INT64, [1]  # Shape of shape is just the rank
    )
    
    # Create the graph
    graph = helper.make_graph(
        [shape1_node, shape2_node],
        "shape_of_shape_model",
        [input_tensor],
        [shape1_output, shape2_output],
    )
    
    # Create the model
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 16)])
    model.ir_version = 8
    
    # Save the model
    file_name = "shape_of_shape.onnx"
    onnx.save(model, file_name)
    print(f"Finished exporting model to {file_name}")
    
    # Compute expected outputs using ReferenceEvaluator
    print("\nComputing expected outputs using ReferenceEvaluator:")
    
    # Create test input
    test_input_shape = (2, 3, 4, 5)
    test_input = np.random.randn(*test_input_shape).astype(np.float32)
    
    # Use ReferenceEvaluator to compute outputs
    sess = ReferenceEvaluator(model)
    outputs = sess.run(None, {"input": test_input})
    
    shape1_output = outputs[0]
    shape2_output = outputs[1]
    
    print(f"Test input shape: {test_input.shape}")
    print(f"shape1 output: {shape1_output}")
    print(f"shape2 output: {shape2_output}")
    
    # Save test data for use in Rust tests
    print(f"\nFor Rust tests:")
    print(f"  Input tensor shape: {list(test_input_shape)}")
    print(f"  shape1 should return: {shape1_output.tolist()}")
    print(f"  shape2 should return: {shape2_output.tolist()}")


if __name__ == "__main__":
    main()