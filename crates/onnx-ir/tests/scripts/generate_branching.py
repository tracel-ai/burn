#!/usr/bin/env -S uv run
# /// script
# dependencies = [
#   "onnx>=1.15.0",
#   "numpy>=1.24.0",
# ]
# ///

"""
Generate ONNX model with branching (multiple consumers).

Tests:
- Single node output consumed by multiple nodes
- Reference counting works correctly
- Rewiring preserves all connections
- Constants aren't incorrectly removed when shared
"""

import onnx
from onnx import helper, TensorProto
import numpy as np


def create_branching_model():
    """Create model where one node's output feeds multiple consumers."""

    # Input
    input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 4])

    # Outputs (3 different outputs from branching)
    output1 = helper.make_tensor_value_info('output1', TensorProto.FLOAT, [1, 4])
    output2 = helper.make_tensor_value_info('output2', TensorProto.FLOAT, [1, 4])
    output3 = helper.make_tensor_value_info('output3', TensorProto.FLOAT, [1, 4])

    # Initializers
    const1 = helper.make_tensor(
        name='const1',
        data_type=TensorProto.FLOAT,
        dims=[1, 4],
        vals=np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32).flatten().tobytes(),
        raw=True
    )

    const2 = helper.make_tensor(
        name='const2',
        data_type=TensorProto.FLOAT,
        dims=[1, 4],
        vals=np.array([[0.5, 0.5, 0.5, 0.5]], dtype=np.float32).flatten().tobytes(),
        raw=True
    )

    # Create nodes with branching structure
    #
    # input → relu → [branch to 3 consumers]
    #                  ├→ add(const1) → output1
    #                  ├→ mul(const2) → output2
    #                  └→ abs → output3
    #
    nodes = [
        # Common node whose output is consumed by multiple nodes
        helper.make_node('Relu', ['input'], ['relu_out'], name='relu'),

        # Consumer 1: Add
        helper.make_node('Add', ['relu_out', 'const1'], ['output1'], name='add'),

        # Consumer 2: Multiply
        helper.make_node('Mul', ['relu_out', 'const2'], ['output2'], name='mul'),

        # Consumer 3: Abs
        helper.make_node('Abs', ['relu_out'], ['output3'], name='abs'),
    ]

    # Create the graph
    graph = helper.make_graph(
        nodes,
        'branching_model',
        [input_tensor],
        [output1, output2, output3],
        initializer=[const1, const2]
    )

    # Create the model
    model = helper.make_model(graph, producer_name="onnx-ir-test", opset_imports=[helper.make_opsetid("", 16)])

    # Check the model
    onnx.checker.check_model(model)

    return model


def main():
    """Generate and save the ONNX model."""
    model = create_branching_model()

    # Save the model
    output_path = '../fixtures/branching.onnx'
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
    print(f"\n  Branching structure:")
    print(f"    relu_out is consumed by: add, mul, abs")


if __name__ == '__main__':
    main()
