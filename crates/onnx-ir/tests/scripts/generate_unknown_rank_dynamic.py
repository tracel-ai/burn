#!/usr/bin/env -S uv run
# /// script
# dependencies = [
#   "onnx>=1.15.0",
#   "numpy>=1.24.0",
# ]
# ///

"""
Generate ONNX model with unknown rank and dynamic shapes.

Tests:
- Type inference with completely unknown tensor rank
- Edge case #41: Unknown rank with dynamic shapes
"""

import onnx
from onnx import helper, TensorProto


def create_unknown_rank_dynamic_model():
    """Create model with unknown rank inputs."""

    # Create input with unspecified shape (unknown rank represented as empty list)
    input1 = helper.make_tensor_value_info('input1', TensorProto.FLOAT, [])  # Unknown shape/rank
    input2 = helper.make_tensor_value_info('input2', TensorProto.FLOAT, ['N', 'C', 'H', 'W'])  # Known rank, dynamic dims

    # Output with partial knowledge
    output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [])

    # Operations that work with unknown ranks
    nodes = [
        # Relu preserves shape regardless of rank
        helper.make_node('Relu', ['input1'], ['temp1'], name='relu'),

        # Add with one unknown rank and one known rank
        helper.make_node('Add', ['temp1', 'input2'], ['output'], name='add'),
    ]

    # Create the graph
    graph = helper.make_graph(
        nodes,
        'unknown_rank_dynamic_model',
        [input1, input2],
        [output],
    )

    # Create the model
    model = helper.make_model(graph, producer_name="onnx-ir-test", opset_imports=[helper.make_opsetid("", 16)])

    # Check the model
    onnx.checker.check_model(model)

    return model


def main():
    """Generate and save the ONNX model."""
    model = create_unknown_rank_dynamic_model()

    # Save the model
    output_path = '../fixtures/unknown_rank_dynamic.onnx'
    onnx.save(model, output_path)
    print(f"Model saved to {output_path}")

    print(f"\nModel info:")
    print(f"  input1: unknown rank (shape=None)")
    print(f"  input2: known rank=4, dynamic dims")
    print(f"  Tests type inference with completely unknown rank")


if __name__ == '__main__':
    main()
