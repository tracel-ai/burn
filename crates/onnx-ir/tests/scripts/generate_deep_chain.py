#!/usr/bin/env -S uv run
# /// script
# dependencies = [
#   "onnx>=1.15.0",
#   "numpy>=1.24.0",
# ]
# ///

"""
Generate ONNX model with a very deep sequential chain of operations.

Tests:
- Type inference convergence with deep graphs
- Graph traversal at scale
- Sequential dependency chains
"""

import onnx
from onnx import helper, TensorProto


def create_deep_chain_model(depth=30):
    """Create model with a deep chain of operations."""

    # Input and output
    input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 4])
    output_tensor = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 4])

    # Create a deep chain: Relu → Abs → Relu → Abs → ... (alternating)
    nodes = []
    current_input = 'input'

    for i in range(depth):
        op_type = 'Relu' if i % 2 == 0 else 'Abs'
        output_name = f'chain_{i}' if i < depth - 1 else 'output'

        nodes.append(
            helper.make_node(op_type, [current_input], [output_name], name=f'{op_type.lower()}_{i}')
        )

        current_input = output_name

    # Create the graph
    graph = helper.make_graph(
        nodes,
        'deep_chain_model',
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
    depth = 30
    model = create_deep_chain_model(depth)

    # Save the model
    output_path = '../fixtures/deep_chain.onnx'
    onnx.save(model, output_path)
    print(f"Model saved to {output_path}")

    print(f"\nModel info:")
    print(f"  Chain depth: {depth} operations")
    print(f"  Pattern: Relu → Abs → Relu → Abs → ...")
    print(f"  Tests type inference convergence with deep graphs")


if __name__ == '__main__':
    main()
