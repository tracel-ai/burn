#!/usr/bin/env python3

# Used to generate model: squeeze_shape_noop.onnx
# Tests Shape(2) -> Shape(2) no-op case by bypassing ONNX validation

import onnx
from onnx import helper, TensorProto
import onnx.shape_inference

def main():
    # Test Shape(2) -> Shape(2) (no-op)
    # We'll create the model without shape inference to bypass validation
    
    # Create input tensor
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [6, 7, 8, 9])
    
    # Shape node to get shape of input (produces [6, 7, 8, 9])
    shape_node = helper.make_node("Shape", ["input"], ["shape_output"])
    
    # Slice to get two dimensions (produces [6, 7])
    starts = helper.make_tensor("starts", TensorProto.INT64, dims=[1], vals=[0])
    ends = helper.make_tensor("ends", TensorProto.INT64, dims=[1], vals=[2])
    slice_node = helper.make_node("Slice", ["shape_output", "starts", "ends"], ["slice_output"])
    
    # Squeeze axis 0 - this will be a no-op since the shape [6, 7] has no dimension of size 1
    squeeze_axes = helper.make_tensor("squeeze_axes", TensorProto.INT64, dims=[1], vals=[0])
    squeeze_node = helper.make_node("Squeeze", ["slice_output", "squeeze_axes"], ["squeeze_output"])
    
    # Output - we expect the same shape [6, 7]
    output = helper.make_tensor_value_info("squeeze_output", TensorProto.INT64, [2])
    
    # Create the graph
    graph = helper.make_graph(
        [shape_node, slice_node, squeeze_node],
        "SqueezeShapeNoOpTest",
        [input_tensor],
        [output],
        [starts, ends, squeeze_axes]
    )
    
    # Create the model
    opset = helper.make_opsetid("", 16)
    model = helper.make_model(graph, opset_imports=[opset])
    
    # Skip shape inference to avoid validation errors
    # onnx.checker.check_model(model, full_check=True)
    
    # Save without validation
    onnx.save(model, "squeeze_shape_noop.onnx")
    
    print("Created squeeze_shape_noop.onnx")
    print("Graph: input tensor -> Shape -> Slice -> Squeeze -> shape output")
    print("Expected: Shape [6,7,8,9] -> [6,7,8,9] -> [6,7] -> [6,7] (no-op)")
    print()
    print("Note: Saved without ONNX validation to test runtime behavior.")

if __name__ == "__main__":
    main()