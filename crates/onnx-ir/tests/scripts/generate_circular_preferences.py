#!/usr/bin/env -S uv run
# /// script
# dependencies = [
#   "onnx>=1.15.0",
#   "numpy>=1.24.0",
# ]
# ///

"""
Generate ONNX model with circular type preferences.

Tests:
- Type inference convergence with circular dependencies
- Edge case #14: Circular preferences convergence
"""

import onnx
from onnx import helper, TensorProto


def create_circular_preferences_model():
    """Create model where type preferences form a cycle."""

    # Inputs with partially specified types (dtype known, shapes dynamic)
    input1 = helper.make_tensor_value_info('input1', TensorProto.FLOAT, ['N', 4])
    input2 = helper.make_tensor_value_info('input2', TensorProto.FLOAT, [2, 'M'])

    # Output
    output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [2, 4])

    # Create a pattern where preferences might cycle:
    # A feeds B, B feeds C, C feeds A (via Add)
    nodes = [
        # First branch
        helper.make_node('Relu', ['input1'], ['branch1'], name='relu1'),

        # Second branch
        helper.make_node('Abs', ['input2'], ['branch2'], name='abs1'),

        # Combine with broadcasting (creates interdependencies)
        helper.make_node('Add', ['branch1', 'branch2'], ['temp1'], name='add1'),

        # Split again
        helper.make_node('Relu', ['temp1'], ['temp2'], name='relu2'),
        helper.make_node('Abs', ['temp1'], ['temp3'], name='abs2'),

        # Merge back
        helper.make_node('Mul', ['temp2', 'temp3'], ['output'], name='mul'),
    ]

    # Create the graph
    graph = helper.make_graph(
        nodes,
        'circular_preferences_model',
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
    model = create_circular_preferences_model()

    # Save the model
    output_path = '../fixtures/circular_preferences.onnx'
    onnx.save(model, output_path)
    print(f"Model saved to {output_path}")

    print(f"\nModel info:")
    print(f"  Complex graph with potential circular preferences")
    print(f"  Tests type inference convergence")


if __name__ == '__main__':
    main()
