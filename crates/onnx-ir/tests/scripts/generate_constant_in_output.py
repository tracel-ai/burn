#!/usr/bin/env -S uv run
# /// script
# dependencies = [
#   "onnx>=1.15.0",
#   "numpy>=1.24.0",
# ]
# ///

"""
Generate ONNX model with constant referenced in graph output.

Tests:
- Constants that are graph outputs (should not be removed)
- Edge case #21: Constant referenced in graph output
"""

import onnx
from onnx import helper, TensorProto
import numpy as np


def create_constant_in_output_model():
    """Create model where a constant is directly in the graph output."""

    # Input
    input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, [2, 3])

    # Outputs
    output1 = helper.make_tensor_value_info('output_computed', TensorProto.FLOAT, [2, 3])
    output2 = helper.make_tensor_value_info('output_constant', TensorProto.FLOAT, [2, 3])

    # Constant that will be in the output
    const_output = helper.make_tensor(
        name='const_for_output',
        data_type=TensorProto.FLOAT,
        dims=[2, 3],
        vals=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32).flatten().tobytes(),
        raw=True
    )

    # Another constant used in computation
    const_scale = helper.make_tensor(
        name='scale',
        data_type=TensorProto.FLOAT,
        dims=[2, 3],
        vals=np.ones((2, 3), dtype=np.float32).flatten().tobytes(),
        raw=True
    )

    # Nodes
    nodes = [
        # Computed output
        helper.make_node('Mul', ['input', 'scale'], ['output_computed'], name='mul'),

        # Constant directly to output (via Identity to make it valid ONNX)
        helper.make_node('Identity', ['const_for_output'], ['output_constant'], name='const_to_output'),
    ]

    # Create the graph
    graph = helper.make_graph(
        nodes,
        'constant_in_output_model',
        [input_tensor],
        [output1, output2],
        initializer=[const_output, const_scale]
    )

    # Create the model
    model = helper.make_model(graph, producer_name="onnx-ir-test", opset_imports=[helper.make_opsetid("", 16)])

    # Check the model
    onnx.checker.check_model(model)

    return model


def main():
    """Generate and save the ONNX model."""
    model = create_constant_in_output_model()

    # Save the model
    output_path = '../fixtures/constant_in_output.onnx'
    onnx.save(model, output_path)
    print(f"Model saved to {output_path}")

    print(f"\nModel info:")
    print(f"  One output is computed, one is a constant")
    print(f"  Constant in output should NOT be removed")


if __name__ == '__main__':
    main()
