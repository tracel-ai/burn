#!/usr/bin/env -S uv run
# /// script
# dependencies = [
#   "onnx>=1.15.0",
#   "numpy>=1.24.0",
# ]
# ///

"""
Generate ONNX model with node that has multiple outputs, but only some are used.

Tests:
- Type inference for unused outputs
- Edge case #28: Node with multiple outputs, only one used
"""

import onnx
from onnx import helper, TensorProto
import numpy as np


def create_node_multiple_outputs_partial_use_model():
    """Create model where a node has multiple outputs but only one is used."""

    # Input
    input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, [3, 4])

    # Output (only one of TopK's two outputs)
    output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [3, 2])

    # K value constant
    k_value = helper.make_tensor(
        name='k',
        data_type=TensorProto.INT64,
        dims=[],
        vals=np.array([2], dtype=np.int64).tobytes(),
        raw=True
    )

    # TopK produces TWO outputs: values and indices
    # But we only use the values output
    nodes = [
        # TopK has 2 outputs: [values, indices]
        # We'll only use 'values' and ignore 'indices'
        helper.make_node(
            'TopK',
            ['input', 'k'],  # In opset 16, K is an input not attribute
            ['topk_values', 'topk_indices'],  # TWO outputs
            name='topk',
            axis=-1
        ),

        # Only use the 'values' output, 'indices' is unused
        helper.make_node('Relu', ['topk_values'], ['output'], name='relu'),
    ]

    # Create the graph
    graph = helper.make_graph(
        nodes,
        'node_multiple_outputs_partial_use_model',
        [input_tensor],
        [output],  # Note: topk_indices is NOT in outputs
        initializer=[k_value]
    )

    # Create the model
    model = helper.make_model(graph, producer_name="onnx-ir-test", opset_imports=[helper.make_opsetid("", 16)])

    # Check the model
    onnx.checker.check_model(model)

    return model


def main():
    """Generate and save the ONNX model."""
    model = create_node_multiple_outputs_partial_use_model()

    # Save the model
    output_path = '../fixtures/node_multiple_outputs_partial_use.onnx'
    onnx.save(model, output_path)
    print(f"Model saved to {output_path}")

    print(f"\nModel info:")
    print(f"  TopK node produces 2 outputs: values, indices")
    print(f"  Only 'values' output is used")
    print(f"  'indices' output is unused (not consumed)")
    print(f"  Tests handling of unused node outputs")


if __name__ == '__main__':
    main()
