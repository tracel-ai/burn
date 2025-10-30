#!/usr/bin/env -S uv run
# /// script
# dependencies = [
#   "onnx>=1.15.0",
#   "numpy>=1.24.0",
# ]
# ///

"""
Generate ONNX model with optional inputs not provided (Clip with only min).

Tests:
- Handling of optional inputs
- Edge case #29: Optional input not provided (Clip)
"""

import onnx
from onnx import helper, TensorProto
import numpy as np


def create_optional_input_clip_model():
    """Create model with Clip that has optional max input not provided."""

    # Input
    input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, [2, 3])

    # Output
    output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [2, 3])

    # Min constant
    min_val = helper.make_tensor(
        name='min_value',
        data_type=TensorProto.FLOAT,
        dims=[],
        vals=np.array([0.0], dtype=np.float32).tobytes(),
        raw=True
    )

    # Clip with only min, no max (max is optional and not provided)
    # Using empty string for max input means "not provided"
    nodes = [
        helper.make_node(
            'Clip',
            ['input', 'min_value', ''],  # Third input (max) is empty = not provided
            ['output'],
            name='clip_optional_max'
        ),
    ]

    # Create the graph
    graph = helper.make_graph(
        nodes,
        'optional_input_clip_model',
        [input_tensor],
        [output],
        initializer=[min_val]
    )

    # Create the model
    model = helper.make_model(graph, producer_name="onnx-ir-test", opset_imports=[helper.make_opsetid("", 16)])

    # Check the model
    onnx.checker.check_model(model)

    return model


def main():
    """Generate and save the ONNX model."""
    model = create_optional_input_clip_model()

    # Save the model
    output_path = '../fixtures/optional_input_clip.onnx'
    onnx.save(model, output_path)
    print(f"Model saved to {output_path}")

    print(f"\nModel info:")
    print(f"  Clip operation with only 'min' provided")
    print(f"  'max' input is optional and NOT provided (empty string)")
    print(f"  Tests handling of optional inputs")


if __name__ == '__main__':
    main()
