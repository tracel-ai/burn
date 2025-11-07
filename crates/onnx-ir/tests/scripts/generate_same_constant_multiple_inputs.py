#!/usr/bin/env -S uv run
# /// script
# dependencies = [
#   "onnx>=1.15.0",
#   "numpy>=1.24.0",
# ]
# ///

"""
Generate ONNX model where ONE operation has multiple input slots referencing THE SAME constant.

Tests:
- Constant lifting when same constant is referenced multiple times in one operation
- Reference counting for single operation with repeated constant inputs
- Edge case: Multiple inputs of one operation → same constant
"""

import onnx
from onnx import helper, TensorProto
import numpy as np


def create_same_constant_multiple_inputs_model():
    """Create model where one operation uses the same constant multiple times."""

    # Input
    input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, [2, 3])

    # Output
    output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [2, 3])

    # A single constant that will be used multiple times BY THE SAME OPERATION
    shared_const = helper.make_tensor(
        name='shared_constant',
        data_type=TensorProto.FLOAT,
        dims=[2, 3],
        vals=np.ones((2, 3), dtype=np.float32).flatten().tobytes(),
        raw=True
    )

    # Boolean condition for Where operation
    bool_condition = helper.make_tensor(
        name='condition',
        data_type=TensorProto.BOOL,
        dims=[2, 3],
        vals=np.ones((2, 3), dtype=bool).flatten().tolist()
    )

    # Operations where the SAME constant appears in multiple input slots
    nodes = [
        # Where: Uses the same constant for BOTH true_value and false_value
        # This is the key test: ONE operation with multiple inputs pointing to SAME constant
        helper.make_node(
            'Where',
            ['condition', 'shared_constant', 'shared_constant'],  # condition, true_val, false_val
            ['output'],
            name='where_same_const'
        ),
    ]

    # Create the graph
    graph = helper.make_graph(
        nodes,
        'same_constant_multiple_inputs_model',
        [input_tensor],
        [output],
        initializer=[shared_const, bool_condition]
    )

    # Create the model
    model = helper.make_model(graph, producer_name="onnx-ir-test", opset_imports=[helper.make_opsetid("", 16)])

    # Check the model
    onnx.checker.check_model(model)

    return model


def main():
    """Generate and save the ONNX model."""
    model = create_same_constant_multiple_inputs_model()

    # Save the model
    output_path = '../fixtures/same_constant_multiple_inputs.onnx'
    onnx.save(model, output_path)
    print(f"Model saved to {output_path}")

    print(f"\nModel info:")
    print(f"  Where operation uses 'shared_constant' for BOTH true_value and false_value")
    print(f"  Tests: ONE operation, MULTIPLE input slots, SAME constant")
    print(f"  Critical edge case for constant lifting and reference counting")


if __name__ == '__main__':
    main()
