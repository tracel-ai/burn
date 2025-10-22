#!/usr/bin/env -S uv run
# /// script
# dependencies = [
#   "onnx>=1.15.0",
#   "numpy>=1.24.0",
# ]
# ///

"""
Generate ONNX model testing constant handling.

Tests:
- Initializers → Constant nodes (Static value source)
- Runtime inputs → Dynamic value source
- Constant lifting to Static arguments
- Unreferenced constant removal
"""

import onnx
from onnx import helper, TensorProto
import numpy as np


def create_constants_model():
    """Create model with various constant handling scenarios."""

    # Runtime input (Dynamic)
    runtime_input = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 3, 4, 4])

    # Output
    output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 3, 4, 4])

    # Initializers (will become Constant nodes, then Static)
    # Used constant - for Add
    add_bias = helper.make_tensor(
        name='add_bias',
        data_type=TensorProto.FLOAT,
        dims=[1, 3, 1, 1],
        vals=np.array([1.0, 2.0, 3.0], dtype=np.float32).tobytes(),
        raw=True
    )

    # Used constant - for Mul
    mul_scale = helper.make_tensor(
        name='mul_scale',
        data_type=TensorProto.FLOAT,
        dims=[1, 3, 1, 1],
        vals=np.array([0.5, 0.5, 0.5], dtype=np.float32).tobytes(),
        raw=True
    )

    # Unused constant (should be removed in Phase 5)
    unused_const = helper.make_tensor(
        name='unused_constant',
        data_type=TensorProto.FLOAT,
        dims=[1, 3, 1, 1],
        vals=np.array([99.0, 99.0, 99.0], dtype=np.float32).tobytes(),
        raw=True
    )

    # Create nodes
    nodes = [
        # Add with initializer (constant will be lifted to Static)
        helper.make_node('Add', ['x', 'add_bias'], ['added'], name='add'),

        # Multiply with initializer
        helper.make_node('Mul', ['added', 'mul_scale'], ['output'], name='mul'),
    ]

    # Create the graph
    graph = helper.make_graph(
        nodes,
        'constants_model',
        [runtime_input],
        [output],
        initializer=[add_bias, mul_scale, unused_const]  # unused_const should be removed
    )

    # Create the model
    model = helper.make_model(graph, producer_name="onnx-ir-test", opset_imports=[helper.make_opsetid("", 16)])

    # Check the model
    onnx.checker.check_model(model)

    return model


def main():
    """Generate and save the ONNX model."""
    model = create_constants_model()

    # Save the model
    output_path = '../fixtures/constants.onnx'
    onnx.save(model, output_path)
    print(f"Model saved to {output_path}")

    # Print model info
    print(f"\nModel info:")
    print(f"  Opset version: {model.opset_import[0].version}")
    print(f"  Inputs: {[inp.name for inp in model.graph.input]}")
    print(f"  Outputs: {[out.name for out in model.graph.output]}")
    print(f"  Initializers: {[init.name for init in model.graph.initializer]}")
    print(f"  Nodes: {len(model.graph.node)}")
    for node in model.graph.node:
        print(f"    - {node.op_type} ({node.name}): {list(node.input)} → {list(node.output)}")


if __name__ == '__main__':
    main()
