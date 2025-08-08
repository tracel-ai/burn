#!/usr/bin/env python3

# Used to generate model: squeeze_shape.onnx

import numpy as np
import onnx
from onnx import helper, TensorProto
from onnx.reference import ReferenceEvaluator

# ONNX opset version to use for model generation
OPSET_VERSION = 16

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
    model = helper.make_model(
        graph, 
        opset_imports=[helper.make_operatorsetid("", OPSET_VERSION)]
    )
    
    # Check and save
    onnx.checker.check_model(model, full_check=True)
    onnx_name = "squeeze_shape.onnx"
    onnx.save(model, onnx_name)
    
    print(f"Created {onnx_name}")
    print("Graph: input tensor -> Shape -> Slice -> Squeeze -> scalar output")
    
    # Test the model with sample data
    test_input = np.random.randn(3, 4, 5).astype(np.float32)
    
    print(f"\nTest input shape: {test_input.shape}")
    
    # Run the model using ReferenceEvaluator
    session = ReferenceEvaluator(onnx_name, verbose=0)
    output = session.run(None, {"input": test_input})
    
    print(f"\nTest output: {repr(output[0])}")
    print()
    print("Note: The Shape(>1) squeeze no-op case is tested in unit tests since")
    print("ONNX's static shape inference is too restrictive for runtime behavior.")

if __name__ == "__main__":
    main()