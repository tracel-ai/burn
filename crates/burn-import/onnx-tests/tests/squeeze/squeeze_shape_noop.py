#!/usr/bin/env python3

# Used to generate model: squeeze_shape_noop.onnx
# Tests Shape(2) -> Shape(2) no-op case by bypassing ONNX validation

import numpy as np
import onnx
from onnx import helper, TensorProto
import onnx.shape_inference
from onnx.reference import ReferenceEvaluator

# ONNX opset version to use for model generation
OPSET_VERSION = 16

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
    model = helper.make_model(
        graph, 
        opset_imports=[helper.make_operatorsetid("", OPSET_VERSION)]
    )
    
    # Skip shape inference to avoid validation errors
    # onnx.checker.check_model(model, full_check=True)
    
    # Save without validation
    onnx_name = "squeeze_shape_noop.onnx"
    onnx.save(model, onnx_name)
    
    print(f"Created {onnx_name}")
    print("Graph: input tensor -> Shape -> Slice -> Squeeze -> shape output")
    
    # Test the model with sample data
    test_input = np.random.randn(6, 7, 8, 9).astype(np.float32)
    
    print(f"\nTest input shape: {test_input.shape}")
    
    # Run the model using ReferenceEvaluator
    try:
        session = ReferenceEvaluator(onnx_name, verbose=0)
        output = session.run(None, {"input": test_input})
        print(f"\nTest output: {repr(output[0])}")
    except ValueError as e:
        print(f"\nExpected error from ReferenceEvaluator: {e}")
        print("This is expected because we're trying to squeeze an axis that doesn't have size 1.")
        print("The Burn runtime handles this as a no-op, returning [6, 7] unchanged.")
    
    print()
    print("Note: Saved without ONNX validation to test runtime behavior.")

if __name__ == "__main__":
    main()