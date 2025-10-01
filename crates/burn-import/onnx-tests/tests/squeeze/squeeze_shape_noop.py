#!/usr/bin/env python3

# Used to generate model: squeeze_shape_noop.onnx
# Tests Burn's squeeze no-op behavior: Shape -> Squeeze(axis=0) -> Shape unchanged
# Demonstrates how Burn handles squeeze operations on Shape types with multiple elements

import numpy as np
import onnx
from onnx import helper, TensorProto
import onnx.shape_inference
from onnx.reference import ReferenceEvaluator

# ONNX opset version to use for model generation
OPSET_VERSION = 16

def main():
    # Create an ONNX model: Shape -> Squeeze(axis=0)
    # Input tensor [6, 7] -> Shape produces [6, 7] -> Squeeze axis 0 -> [6, 7] unchanged
    # This tests Burn's handling of squeeze no-op when the Shape has multiple elements

    # Create input tensor with 2D shape
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [6, 7])

    # Shape node to get shape of input (produces [6, 7])
    shape_node = helper.make_node("Shape", ["input"], ["shape_output"])

    # Squeeze axis 0 on the shape output
    # In standard ONNX, this would fail since [6, 7] has axis 0 with size 6, not 1
    # But Burn treats this as a no-op, keeping Shape([6, 7]) unchanged
    squeeze_axes = helper.make_tensor("squeeze_axes", TensorProto.INT64, dims=[1], vals=[0])
    squeeze_node = helper.make_node("Squeeze", ["shape_output", "squeeze_axes"], ["squeeze_output"])

    # Output - Burn will keep this as Shape([6, 7]) since squeeze is a no-op
    output = helper.make_tensor_value_info("squeeze_output", TensorProto.INT64, [2])

    # Create the graph
    graph = helper.make_graph(
        [shape_node, squeeze_node],
        "SqueezeShapeNoOpTest",
        [input_tensor],
        [output],
        [squeeze_axes]
    )

    # Create the model
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_operatorsetid("", OPSET_VERSION)]
    )

    # Note: This model might not pass strict ONNX validation in some implementations
    # But Burn handles it gracefully as a no-op
    try:
        onnx.checker.check_model(model, full_check=True)
        print("Model passed ONNX validation")
    except Exception as e:
        print(f"ONNX validation note: {e}")
        print("This is expected - Burn handles this case gracefully")
    
    # Save the model for Burn to handle
    onnx_name = "squeeze_shape_noop.onnx"
    onnx.save(model, onnx_name)

    print(f"Created {onnx_name}")
    print("Graph: input[6,7] -> Shape -> Squeeze(axis=0) -> Shape([6,7]) unchanged")

    # Test the model with sample data
    test_input = np.random.randn(6, 7).astype(np.float32)

    print(f"\nTest input shape: {test_input.shape}")

    # Run the model using ReferenceEvaluator
    # This may or may not work depending on the ONNX implementation
    try:
        session = ReferenceEvaluator(model, verbose=0)
        output, = session.run(None, {"input": test_input})
        print(f"\nReferenceEvaluator output: {repr(output)}")
        print(f"Output shape: {output.shape}")
        print(f"Expected value: [6, 7], Actual: {output}")
    except Exception as e:
        print(f"\nReferenceEvaluator error: {e}")
        print("This is expected - Burn's runtime handles this case as a no-op")
        print("Expected Burn output: Shape([6, 7]) unchanged")

    print()
    print("Note: This tests Burn's squeeze no-op handling for Shape types.")
    print("When squeezing a Shape with multiple elements, Burn keeps it unchanged.")

if __name__ == "__main__":
    main()
