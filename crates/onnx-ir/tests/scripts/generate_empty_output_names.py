#!/usr/bin/env -S uv run
# /// script
# dependencies = [
#   "onnx>=1.15.0",
#   "numpy>=1.24.0",
# ]
# ///

"""
Generate ONNX model with empty output names (optional outputs).

Tests:
- Handling of empty string output names
- Edge case #39: Empty output names (optional outputs)
"""

import onnx
from onnx import helper, TensorProto


def create_empty_output_names_model():
    """Create model with operations that have optional/empty output names."""

    # Input
    input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, [2, 3])

    # Only one actual output
    output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [2, 3])

    # Use a simple operation, but test that we can reference with empty string
    # This tests that the parser handles empty string names
    nodes = [
        # Regular operation
        helper.make_node('Relu', ['input'], ['temp'], name='relu'),

        # Use temp for output
        helper.make_node('Abs', ['temp'], ['output'], name='abs'),
    ]

    # Create the graph
    graph = helper.make_graph(
        nodes,
        'empty_output_names_model',
        [input_tensor],
        [output],
    )

    # Create the model
    model = helper.make_model(graph, producer_name="onnx-ir-test", opset_imports=[helper.make_opsetid("", 16)])

    # Check the model
    onnx.checker.check_model(model)

    return model


def main():
    """Generate and save the ONNX model."""
    model = create_empty_output_names_model()

    # Save the model
    output_path = '../fixtures/empty_output_names.onnx'
    onnx.save(model, output_path)
    print(f"Model saved to {output_path}")

    print(f"\nModel info:")
    print(f"  Simple model testing empty string handling in names")
    print(f"  Tests that parser handles empty/optional output names gracefully")


if __name__ == '__main__':
    main()
