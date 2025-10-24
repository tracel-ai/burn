#!/usr/bin/env -S uv run
# /// script
# dependencies = [
#   "onnx>=1.15.0",
#   "numpy>=1.24.0",
# ]
# ///

"""
Generate ONNX model with nodes NOT in topological order.

Tests:
- Parser's ability to handle non-topologically sorted node lists
- Correctness validation: does the IR maintain correct execution order?
"""

import onnx
from onnx import helper, TensorProto


def create_non_topological_order_model():
    """Create model where ONNX nodes are NOT in topological order."""

    # Input
    input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, [2, 3])

    # Output
    output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [2, 3])

    # Create nodes in REVERSE topological order (intentionally wrong order)
    # Correct order would be: relu → abs → add
    # We'll list them as: add → abs → relu
    nodes = [
        # Node 3: This uses 'abs_out' which is defined later (non-topological!)
        helper.make_node('Add', ['abs_out', 'abs_out'], ['output'], name='add'),

        # Node 2: This uses 'relu_out' which is defined later
        helper.make_node('Abs', ['relu_out'], ['abs_out'], name='abs'),

        # Node 1: This is the actual starting point
        helper.make_node('Relu', ['input'], ['relu_out'], name='relu'),
    ]

    # Create the graph (nodes are in wrong order!)
    graph = helper.make_graph(
        nodes,
        'non_topological_order_model',
        [input_tensor],
        [output],
    )

    # Create the model
    model = helper.make_model(graph, producer_name="onnx-ir-test", opset_imports=[helper.make_opsetid("", 16)])

    # NOTE: We skip check_model() because ONNX checker enforces topological order
    # We want to test if our parser can handle non-topological order
    # onnx.checker.check_model(model)

    return model


def main():
    """Generate and save the ONNX model."""
    model = create_non_topological_order_model()

    # Save the model
    output_path = '../fixtures/non_topological_order.onnx'
    onnx.save(model, output_path)
    print(f"Model saved to {output_path}")

    print(f"\nModel info:")
    print(f"  Nodes are in REVERSE topological order:")
    print(f"    ONNX order: Add → Abs → Relu")
    print(f"    Correct execution order: Relu → Abs → Add")
    print(f"  Tests parser's ability to handle non-topological node lists")


if __name__ == '__main__':
    main()
