#!/usr/bin/env -S uv run
# /// script
# dependencies = [
#   "onnx>=1.15.0",
#   "numpy>=1.24.0",
# ]
# ///

"""
Generate ONNX model to test Static AND Constant value source invariant.

Tests:
- Invariant: A value cannot be both Static (embedded) AND Constant (node reference)
- Edge case #33: Static AND Constant invariant
"""

import onnx
from onnx import helper, TensorProto
import numpy as np


def create_static_and_constant_invariant_model():
    """Create model that tests the Static/Constant value source distinction."""

    # Input
    input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, [2, 3])

    # Outputs
    output1 = helper.make_tensor_value_info('output1', TensorProto.FLOAT, [2, 3])
    output2 = helper.make_tensor_value_info('output2', TensorProto.FLOAT, [2, 3])

    # Static constant (embedded in initializer, will have data_id)
    static_const = helper.make_tensor(
        name='static_value',
        data_type=TensorProto.FLOAT,
        dims=[2, 3],
        vals=np.ones((2, 3), dtype=np.float32).flatten().tobytes(),
        raw=True
    )

    # This constant will be lifted to a Constant node
    const_for_node = helper.make_tensor(
        name='const_for_node',
        data_type=TensorProto.FLOAT,
        dims=[2, 3],
        vals=np.full((2, 3), 2.0, dtype=np.float32).flatten().tobytes(),
        raw=True
    )

    # Operations
    nodes = [
        # Use static constant - should have ValueSource::Constant with data_id
        helper.make_node('Add', ['input', 'static_value'], ['output1'], name='add_static'),

        # Use another constant that goes through Constant node
        helper.make_node('Mul', ['input', 'const_for_node'], ['output2'], name='mul_const'),
    ]

    # Create the graph
    graph = helper.make_graph(
        nodes,
        'static_and_constant_invariant_model',
        [input_tensor],
        [output1, output2],
        initializer=[static_const, const_for_node]
    )

    # Create the model
    model = helper.make_model(graph, producer_name="onnx-ir-test", opset_imports=[helper.make_opsetid("", 16)])

    # Check the model
    onnx.checker.check_model(model)

    return model


def main():
    """Generate and save the ONNX model."""
    model = create_static_and_constant_invariant_model()

    # Save the model
    output_path = '../fixtures/static_constant_invariant.onnx'
    onnx.save(model, output_path)
    print(f"Model saved to {output_path}")

    print(f"\nModel info:")
    print(f"  Tests Static vs Constant value source distinction")
    print(f"  Invariant: A value cannot be BOTH Static AND Constant")


if __name__ == '__main__':
    main()
