#!/usr/bin/env python3

# This script generates an ONNX model that tests the Slice operation with axes parameter.
# The model slices a specific dimension (axis=1) rather than the default first dimension.

import numpy as np
import onnx
from onnx import helper, TensorProto

def create_slice_axes_model():
    """
    Creates an ONNX model that uses Slice with axes parameter.
    Input shape: [2, 4, 6]
    Slice on axis 1: starts=[1], ends=[3], axes=[1]
    Output shape: [2, 2, 6]

    This tests that the axes parameter is correctly handled.
    Without axes, slice would operate on dimension 0.
    With axes=[1], slice operates on dimension 1.
    """

    # Create input
    input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, [2, 4, 6])
    output_tensor = helper.make_tensor_value_info('output', TensorProto.FLOAT, [2, 2, 6])

    # Create slice parameters as initializers
    starts = helper.make_tensor('starts', TensorProto.INT64, [1], [1])
    ends = helper.make_tensor('ends', TensorProto.INT64, [1], [3])
    axes = helper.make_tensor('axes', TensorProto.INT64, [1], [1])  # Slice on axis 1

    # Create the Slice node
    slice_node = helper.make_node(
        'Slice',
        inputs=['input', 'starts', 'ends', 'axes'],
        outputs=['output']
    )

    # Create the graph
    graph = helper.make_graph(
        [slice_node],
        'slice_axes',
        [input_tensor],
        [output_tensor],
        [starts, ends, axes]  # Add as initializers
    )

    # Create the model
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 16)])

    # Check the model
    onnx.checker.check_model(model)

    return model

def main():
    # Create the model
    model = create_slice_axes_model()

    # Save the model
    onnx.save(model, "slice_axes.onnx")
    print("Model saved to slice_axes.onnx")

    # Use ONNX ReferenceEvaluator to verify the model
    from onnx.reference import ReferenceEvaluator

    # Create evaluator
    sess = ReferenceEvaluator("slice_axes.onnx")

    # Get input shape from the model
    input_shape = [d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim]

    # Create test input
    test_input = np.random.randn(*input_shape).astype(np.float32)

    # Run inference
    output = sess.run(None, {'input': test_input})[0]

    # Extract slice parameters from the model
    initializers = {init.name: init for init in model.graph.initializer}
    starts_val = initializers['starts'].int64_data[0]
    ends_val = initializers['ends'].int64_data[0]
    axes_val = initializers['axes'].int64_data[0]

    print(f"\nModel details (verified with ONNX ReferenceEvaluator):")
    print(f"- Input shape: {list(test_input.shape)}")
    print(f"- Slice parameters: starts=[{starts_val}], ends=[{ends_val}], axes=[{axes_val}]")
    print(f"- Output shape: {list(output.shape)}")
    print(f"- Effect: Slices dimension {axes_val} from index {starts_val} to {ends_val}")

    # Verify the slicing behavior
    expected_output = test_input[:, starts_val:ends_val, :]
    np.testing.assert_allclose(output, expected_output, rtol=1e-5)
    print(f"\n✓ Verified: Output matches numpy slice [:, {starts_val}:{ends_val}, :]")
    print("\nThis tests that axes parameter correctly specifies which dimension to slice.")

if __name__ == "__main__":
    main()
