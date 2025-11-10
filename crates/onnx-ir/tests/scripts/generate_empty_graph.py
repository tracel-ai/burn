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
    """Create model where input connects directly to output with ZERO operations."""

    # Input and output are the SAME tensor - no transformation needed
    # This creates a truly empty graph with no nodes
    input_tensor = helper.make_tensor_value_info('data', TensorProto.FLOAT, [1, 4])
    output_tensor = helper.make_tensor_value_info('data', TensorProto.FLOAT, [1, 4])

    # ZERO nodes - input IS the output
    nodes = []

    # Create the graph with no nodes
    graph = helper.make_graph(
        nodes,
        'empty_graph_model',
        [input_tensor],
        [output_tensor],
    )

    # Create the model
    model = helper.make_model(graph, producer_name="onnx-ir-test", opset_imports=[helper.make_opsetid("", 16)])

    # Check the model - this is valid in ONNX!
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
    print(f"  Nodes: {len(model.graph.node)} (ZERO nodes)")
    print(f"  Input tensor 'data' is directly the output tensor 'data'")
    print(f"  Tests absolute minimal graph structure with no operations")


if __name__ == '__main__':
    main()
