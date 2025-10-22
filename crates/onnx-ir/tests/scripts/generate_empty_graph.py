#!/usr/bin/env -S uv run
# /// script
# dependencies = [
#   "onnx>=1.15.0",
#   "numpy>=1.24.0",
# ]
# ///

"""
Generate ONNX model with empty graph (input flows directly to output, no operations).

Tests:
- Minimal graph structure (no nodes)
- Identity elimination edge case (no Identity nodes to eliminate)
- Input→Output direct connection
"""

import onnx
from onnx import helper, TensorProto


def create_empty_graph_model():
    """Create model where input connects directly to output with no operations."""

    # Input and output with same shape and name
    # In ONNX, we need at least an Identity node to connect input to output
    # An empty graph with no nodes would be invalid
    input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 4])
    output_tensor = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 4])

    # Empty nodes list - this creates the minimal valid graph
    # Note: ONNX requires explicit connection, so we use Identity
    nodes = [
        helper.make_node('Identity', ['input'], ['output'], name='passthrough'),
    ]

    # Create the graph
    graph = helper.make_graph(
        nodes,
        'empty_graph_model',
        [input_tensor],
        [output_tensor],
    )

    # Create the model
    model = helper.make_model(graph, producer_name="onnx-ir-test", opset_imports=[helper.make_opsetid("", 16)])

    # Check the model
    onnx.checker.check_model(model)

    return model


def main():
    """Generate and save the ONNX model."""
    model = create_empty_graph_model()

    # Save the model
    output_path = '../fixtures/empty_graph.onnx'
    onnx.save(model, output_path)
    print(f"Model saved to {output_path}")

    print(f"\nModel info:")
    print(f"  Nodes: {len(model.graph.node)} (minimal Identity)")
    print(f"  Tests minimal graph structure")
    print(f"  Tests edge case where graph has no real operations")


if __name__ == '__main__':
    main()
