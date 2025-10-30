#!/usr/bin/env -S uv run
# /// script
# dependencies = [
#   "onnx>=1.15.0",
#   "numpy>=1.24.0",
# ]
# ///

"""
Generate ONNX model with diamond pattern (split then merge).

Tests:
- Type inference with split and merge paths
- Convergence when multiple paths reconverge
- Edge case #9: Diamond pattern (split then merge)
"""

import onnx
from onnx import helper, TensorProto


def create_diamond_pattern_model():
    """Create model where computation splits and then merges back."""

    # Input
    input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 4])

    # Output
    output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 4])

    # Diamond pattern:
    #     input
    #     /   \
    #  Relu  Abs
    #     \   /
    #      Add  -> output

    nodes = [
        # Split: input feeds two different operations
        helper.make_node('Relu', ['input'], ['path1'], name='relu_path'),
        helper.make_node('Abs', ['input'], ['path2'], name='abs_path'),

        # Merge: both paths combine
        helper.make_node('Add', ['path1', 'path2'], ['output'], name='merge'),
    ]

    # Create the graph
    graph = helper.make_graph(
        nodes,
        'diamond_pattern_model',
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
    model = create_diamond_pattern_model()

    # Save the model
    output_path = '../fixtures/diamond_pattern.onnx'
    onnx.save(model, output_path)
    print(f"Model saved to {output_path}")

    print(f"\nModel info:")
    print(f"  Diamond pattern: input → (Relu, Abs) → Add → output")
    print(f"  Tests split and merge convergence")


if __name__ == '__main__':
    main()
