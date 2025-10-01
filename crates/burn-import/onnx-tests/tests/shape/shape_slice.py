#!/usr/bin/env python3

# used to generate model: onnx-tests/tests/shape/shape_slice.onnx
# This tests the Shape operation with start/end parameters

import onnx
from onnx import helper, TensorProto
from onnx.reference import ReferenceEvaluator
import numpy as np


def main():
    """
    Create an ONNX model that uses Shape operation with start and end attributes.
    This tests slicing of shape dimensions.
    """
    
    # Create input tensor placeholder
    input_tensor = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [2, 3, 4, 5, 6]
    )
    
    # Shape node with start=1, end=4 (should extract dims [3, 4, 5])
    shape_slice_node = helper.make_node(
        "Shape",
        inputs=["input"],
        outputs=["shape_slice"],
        name="shape_slice_node",
        start=1,
        end=4
    )
    
    # Create output
    shape_output = helper.make_tensor_value_info(
        "shape_slice", TensorProto.INT64, [3]  # Extracting 3 dimensions
    )
    
    # Create the graph
    graph = helper.make_graph(
        [shape_slice_node],
        "shape_slice_model",
        [input_tensor],
        [shape_output],
    )
    
    # Create the model
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 16)])
    model.ir_version = 8
    
    # Save the model
    file_name = "shape_slice.onnx"
    onnx.save(model, file_name)
    print(f"Finished exporting model to {file_name}")
    
    # Compute expected outputs using ReferenceEvaluator
    print("\nComputing expected outputs using ReferenceEvaluator:")
    
    # Create test input
    test_input_shape = (2, 3, 4, 5, 6)
    test_input = np.random.randn(*test_input_shape).astype(np.float32)
    
    # Use ReferenceEvaluator to compute output
    sess = ReferenceEvaluator(model)
    outputs = sess.run(None, {"input": test_input})
    
    shape_slice_output = outputs[0]
    
    print(f"Test input shape: {test_input.shape}")
    print(f"Shape slice output (start=1, end=4): {shape_slice_output}")
    
    # Save test data for use in Rust tests
    print(f"\nFor Rust tests:")
    print(f"  Input tensor shape: {list(test_input_shape)}")
    print(f"  With start=1, end=4")
    print(f"  Should return: {shape_slice_output.tolist()}")


if __name__ == "__main__":
    main()