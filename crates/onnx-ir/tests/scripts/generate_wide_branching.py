#!/usr/bin/env -S uv run
# /// script
# dependencies = [
#   "onnx>=1.15.0",
#   "numpy>=1.24.0",
# ]
# ///

"""
Generate ONNX model with wide branching (one node feeding many consumers).

Tests:
- Reference counting with many consumers
- Connectivity at scale
- Wide dependency fan-out
"""

import onnx
from onnx import helper, TensorProto
import numpy as np


def create_wide_branching_model(num_outputs=8):
    """Create model where one node's output feeds many consumers."""

    # Single input
    input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 4])

    # Many outputs
    outputs = [
        helper.make_tensor_value_info(f'output{i}', TensorProto.FLOAT, [1, 4])
        for i in range(num_outputs)
    ]

    # Initializers for operations
    initializers = []
    for i in range(num_outputs):
        init = helper.make_tensor(
            name=f'const{i}',
            data_type=TensorProto.FLOAT,
            dims=[1, 4],
            vals=np.array([[i + 1.0] * 4], dtype=np.float32).flatten().tobytes(),
            raw=True
        )
        initializers.append(init)

    # Nodes: Single Relu branches to many consumers
    nodes = [
        # Single node that will be consumed by many
        helper.make_node('Relu', ['input'], ['relu_out'], name='relu'),
    ]

    # Each consumer adds the relu_out with a different constant
    for i in range(num_outputs):
        nodes.append(
            helper.make_node(
                'Add',
                ['relu_out', f'const{i}'],
                [f'output{i}'],
                name=f'add{i}'
            )
        )

    # Create the graph
    graph = helper.make_graph(
        nodes,
        'wide_branching_model',
        [input_tensor],
        outputs,
        initializer=initializers
    )

    # Create the model
    model = helper.make_model(graph, producer_name="onnx-ir-test", opset_imports=[helper.make_opsetid("", 16)])

    # Check the model
    onnx.checker.check_model(model)

    return model


def main():
    """Generate and save the ONNX model."""
    num_outputs = 8
    model = create_wide_branching_model(num_outputs)

    # Save the model
    output_path = '../fixtures/wide_branching.onnx'
    onnx.save(model, output_path)
    print(f"Model saved to {output_path}")

    print(f"\nModel info:")
    print(f"  1 input → Relu → {num_outputs} outputs")
    print(f"  Single node consumed by {num_outputs} different operations")
    print(f"  Tests reference counting at scale")


if __name__ == '__main__':
    main()
