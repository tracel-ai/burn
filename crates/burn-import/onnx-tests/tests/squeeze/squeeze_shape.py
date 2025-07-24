#!/usr/bin/env python3

# Used to generate model: squeeze_shape.onnx

import numpy as np
import onnx
from onnx import helper, TensorProto

def main():
    # Test Shape(1) -> Scalar
    # Create input tensor
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [3, 4, 5])
    
    # Shape node to get shape of input (produces [3, 4, 5])
    shape_node = helper.make_node("Shape", ["input"], ["shape_output"])
    
    # Slice to get a single dimension (produces [3])
    starts = helper.make_tensor("starts", TensorProto.INT64, dims=[1], vals=[0])
    ends = helper.make_tensor("ends", TensorProto.INT64, dims=[1], vals=[1])
    slice_node = helper.make_node("Slice", ["shape_output", "starts", "ends"], ["slice_output"])
    
    # Squeeze the shape from [3] to scalar 3
    squeeze_axes = helper.make_tensor("squeeze_axes", TensorProto.INT64, dims=[1], vals=[0])
    squeeze_node = helper.make_node("Squeeze", ["slice_output", "squeeze_axes"], ["squeeze_output"])
    
    # Output
    output = helper.make_tensor_value_info("squeeze_output", TensorProto.INT64, [])  # scalar output
    
    # Create the graph
    graph = helper.make_graph(
        [shape_node, slice_node, squeeze_node],
        "SqueezeShapeTest",
        [input_tensor],
        [output],
        [starts, ends, squeeze_axes]
    )
    
    # Create the model
    opset = helper.make_opsetid("", 16)
    model = helper.make_model(graph, opset_imports=[opset])
    
    # Check and save
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, "squeeze_shape.onnx")
    
    print("Created squeeze_shape.onnx")
    print("Graph: input tensor -> Shape -> Slice -> Squeeze -> scalar output")
    print("Expected: Shape [3,4,5] -> [3,4,5] -> [3] -> 3")
    print()
    print("Note: The Shape(>1) squeeze no-op case is tested in unit tests since")
    print("ONNX's static shape inference is too restrictive for runtime behavior.")

if __name__ == "__main__":
    main()