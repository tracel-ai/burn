#!/usr/bin/env -S uv run
# /// script
# dependencies = [
#   "onnx>=1.15.0",
#   "numpy>=1.24.0",
# ]
# ///

"""
Generate ONNX model that tests broadcasting edge cases.

Tests:
- Compatible broadcasting that requires careful shape inference
- Edge case #11: Broadcasting with shapes that need alignment
"""

import onnx
from onnx import helper, TensorProto
import numpy as np


def create_incompatible_broadcast_model():
    """Create model with broadcasting that tests shape inference."""

    # Inputs with different shapes that can broadcast
    input1 = helper.make_tensor_value_info('input1', TensorProto.FLOAT, [1, 3, 1, 4])  # [1,3,1,4]
    input2 = helper.make_tensor_value_info('input2', TensorProto.FLOAT, [2, 1, 5, 1])  # [2,1,5,1]

    # Output should be [2,3,5,4] after broadcasting
    output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [2, 3, 5, 4])

    # Constants with different broadcastable shapes
    const1 = helper.make_tensor(
        name='const1',
        data_type=TensorProto.FLOAT,
        dims=[3, 1, 1],  # [3,1,1] broadcasts with [1,3,1,4]
        vals=np.array([[[1.0]], [[2.0]], [[3.0]]], dtype=np.float32).flatten().tobytes(),
        raw=True
    )

    # Nodes testing broadcasting
    nodes = [
        # Add with different ranks
        helper.make_node('Add', ['input1', 'input2'], ['temp1'], name='add_broadcast'),

        # Multiply with constant of different shape
        helper.make_node('Mul', ['temp1', 'const1'], ['output'], name='mul_broadcast'),
    ]

    # Create the graph
    graph = helper.make_graph(
        nodes,
        'incompatible_broadcast_model',
        [input1, input2],
        [output],
        initializer=[const1]
    )

    # Create the model
    model = helper.make_model(graph, producer_name="onnx-ir-test", opset_imports=[helper.make_opsetid("", 16)])

    # Check the model
    onnx.checker.check_model(model)

    return model


def main():
    """Generate and save the ONNX model."""
    model = create_incompatible_broadcast_model()

    # Save the model
    output_path = '../fixtures/complex_broadcasting.onnx'
    onnx.save(model, output_path)
    print(f"Model saved to {output_path}")

    print(f"\nModel info:")
    print(f"  Broadcasting: [1,3,1,4] + [2,1,5,1] = [2,3,5,4]")
    print(f"  Tests complex NumPy-style broadcasting")


if __name__ == '__main__':
    main()
