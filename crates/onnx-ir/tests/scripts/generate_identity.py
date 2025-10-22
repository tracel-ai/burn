#!/usr/bin/env -S uv run
# /// script
# dependencies = [
#   "onnx>=1.15.0",
#   "numpy>=1.24.0",
# ]
# ///

"""
Generate ONNX model testing Identity node elimination.

Tests:
- Chain of Identity nodes (A → Identity → Identity → B)
- Transitive rewiring in Phase 4
- Graph output rewiring
- Identity elimination preserves connectivity
"""

import onnx
from onnx import helper, TensorProto
import numpy as np


def create_identity_model():
    """Create model with Identity nodes that should be eliminated."""

    # Input
    input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 4])

    # Output
    output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 4])

    # Initializer
    const = helper.make_tensor(
        name='const',
        data_type=TensorProto.FLOAT,
        dims=[1, 4],
        vals=np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32).flatten().tobytes(),
        raw=True
    )

    # Create nodes with Identity chain
    #
    # input → relu → identity1 → identity2 → add(const) → identity3 → output
    #
    # After Phase 4, identities should be eliminated:
    # input → relu → add(const) → output
    #
    nodes = [
        # Real operation
        helper.make_node('Relu', ['input'], ['relu_out'], name='relu'),

        # Identity chain (should be eliminated)
        helper.make_node('Identity', ['relu_out'], ['identity1_out'], name='identity1'),
        helper.make_node('Identity', ['identity1_out'], ['identity2_out'], name='identity2'),

        # Real operation
        helper.make_node('Add', ['identity2_out', 'const'], ['add_out'], name='add'),

        # Identity before output (should be eliminated, output rewired)
        helper.make_node('Identity', ['add_out'], ['output'], name='identity3'),
    ]

    # Create the graph
    graph = helper.make_graph(
        nodes,
        'identity_model',
        [input_tensor],
        [output],
        initializer=[const]
    )

    # Create the model
    model = helper.make_model(graph, producer_name='onnx-ir-test')
    model.opset_import[0].version = 16

    # Check the model
    onnx.checker.check_model(model)

    return model


def main():
    """Generate and save the ONNX model."""
    model = create_identity_model()

    # Save the model
    output_path = '../fixtures/identity.onnx'
    onnx.save(model, output_path)
    print(f"Model saved to {output_path}")

    # Print model info
    print(f"\nModel info:")
    print(f"  Opset version: {model.opset_import[0].version}")
    print(f"  Inputs: {[inp.name for inp in model.graph.input]}")
    print(f"  Outputs: {[out.name for out in model.graph.output]}")
    print(f"  Nodes: {len(model.graph.node)}")
    for node in model.graph.node:
        print(f"    - {node.op_type} ({node.name}): {list(node.input)} → {list(node.output)}")
    print(f"\n  Expected after Phase 4:")
    print(f"    - Identity nodes should be eliminated")
    print(f"    - Only Relu and Add nodes should remain")


if __name__ == '__main__':
    main()
