#!/usr/bin/env -S uv run
# /// script
# dependencies = [
#   "onnx>=1.15.0",
#   "numpy>=1.24.0",
# ]
# ///

"""
Generate ONNX model with a constant referenced multiple times.

Tests:
- Reference counting for constants used multiple times
- Edge case #20: Constant referenced multiple times
"""

import onnx
from onnx import helper, TensorProto
import numpy as np


def create_constant_multiple_refs_model():
    """Create model where one constant is used by multiple operations."""

    # Input
    input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, [2, 3])

    # Outputs
    output1 = helper.make_tensor_value_info('output1', TensorProto.FLOAT, [2, 3])
    output2 = helper.make_tensor_value_info('output2', TensorProto.FLOAT, [2, 3])
    output3 = helper.make_tensor_value_info('output3', TensorProto.FLOAT, [2, 3])

    # A single constant used multiple times
    shared_const = helper.make_tensor(
        name='shared_constant',
        data_type=TensorProto.FLOAT,
        dims=[2, 3],
        vals=np.ones((2, 3), dtype=np.float32).flatten().tobytes(),
        raw=True
    )

    # Multiple operations using the same constant
    nodes = [
        # All three operations use the same constant
        helper.make_node('Add', ['input', 'shared_constant'], ['output1'], name='add1'),
        helper.make_node('Mul', ['input', 'shared_constant'], ['output2'], name='mul1'),
        helper.make_node('Sub', ['shared_constant', 'input'], ['output3'], name='sub1'),
    ]

    # Create the graph
    graph = helper.make_graph(
        nodes,
        'constant_multiple_refs_model',
        [input_tensor],
        [output1, output2, output3],
        initializer=[shared_const]
    )

    # Create the model
    model = helper.make_model(graph, producer_name="onnx-ir-test", opset_imports=[helper.make_opsetid("", 16)])

    # Check the model
    onnx.checker.check_model(model)

    return model


def main():
    """Generate and save the ONNX model."""
    model = create_constant_multiple_refs_model()

    # Save the model
    output_path = '../fixtures/constant_multiple_refs.onnx'
    onnx.save(model, output_path)
    print(f"Model saved to {output_path}")

    print(f"\nModel info:")
    print(f"  One constant used by 3 different operations")
    print(f"  Tests reference counting with multiple consumers")


if __name__ == '__main__':
    main()
