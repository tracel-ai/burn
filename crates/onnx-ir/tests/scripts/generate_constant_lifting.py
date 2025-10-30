#!/usr/bin/env -S uv run
# /// script
# dependencies = [
#   "onnx>=1.15.0",
#   "numpy>=1.24.0",
# ]
# ///

"""
Generate ONNX model testing constant lifting in Phase 2.

Tests:
- Constants lifted during node conversion
- Edge case #15: Constants lifted in Phase 2, used in Phase 3
"""

import onnx
from onnx import helper, TensorProto
import numpy as np


def create_constant_lifting_model():
    """Create model where constants are lifted during conversion."""

    # Input
    input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, [2, 3])

    # Output
    output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [2, 3])

    # Initializers that will be lifted as constants
    weight = helper.make_tensor(
        name='weight',
        data_type=TensorProto.FLOAT,
        dims=[3, 3],
        vals=np.random.randn(3, 3).astype(np.float32).tobytes(),
        raw=True
    )

    bias = helper.make_tensor(
        name='bias',
        data_type=TensorProto.FLOAT,
        dims=[3],
        vals=np.random.randn(3).astype(np.float32).tobytes(),
        raw=True
    )

    scale = helper.make_tensor(
        name='scale',
        data_type=TensorProto.FLOAT,
        dims=[3],
        vals=np.array([1.0, 2.0, 3.0], dtype=np.float32).tobytes(),
        raw=True
    )

    # Operations that use lifted constants
    nodes = [
        # MatMul uses weight constant (lifted in Phase 2)
        helper.make_node('MatMul', ['input', 'weight'], ['temp1'], name='matmul'),

        # Add uses bias constant
        helper.make_node('Add', ['temp1', 'bias'], ['temp2'], name='add_bias'),

        # Mul uses scale constant
        helper.make_node('Mul', ['temp2', 'scale'], ['output'], name='mul_scale'),
    ]

    # Create the graph
    graph = helper.make_graph(
        nodes,
        'constant_lifting_model',
        [input_tensor],
        [output],
        initializer=[weight, bias, scale]
    )

    # Create the model
    model = helper.make_model(graph, producer_name="onnx-ir-test", opset_imports=[helper.make_opsetid("", 16)])

    # Check the model
    onnx.checker.check_model(model)

    return model


def main():
    """Generate and save the ONNX model."""
    model = create_constant_lifting_model()

    # Save the model
    output_path = '../fixtures/constant_lifting.onnx'
    onnx.save(model, output_path)
    print(f"Model saved to {output_path}")

    print(f"\nModel info:")
    print(f"  Multiple initializers used as constants")
    print(f"  Tests constant lifting in Phase 2")


if __name__ == '__main__':
    main()
