#!/usr/bin/env -S uv run
# /// script
# dependencies = [
#   "onnx>=1.15.0",
#   "numpy>=1.24.0",
# ]
# ///

"""
Generate ONNX model that tests type validation and handling.

Tests:
- Mixed scalar and tensor operations
- Type inference with ambiguous shapes
- Handling of scalar vs tensor distinction in operations
"""

import onnx
from onnx import helper, TensorProto
import numpy as np


def create_type_mismatch_model():
    """Create model with operations that test type inference edge cases."""

    # Inputs with different characteristics
    input_tensor = helper.make_tensor_value_info('input_tensor', TensorProto.FLOAT, [2, 3])
    input_scalar = helper.make_tensor_value_info('input_scalar', TensorProto.FLOAT, [])  # Scalar

    # Output
    output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [2, 3])

    # Constants with different types
    const_scalar = helper.make_tensor(
        name='const_scalar',
        data_type=TensorProto.FLOAT,
        dims=[],  # Scalar
        vals=np.array([2.0], dtype=np.float32).tobytes(),
        raw=True
    )

    const_tensor = helper.make_tensor(
        name='const_tensor',
        data_type=TensorProto.FLOAT,
        dims=[2, 3],
        vals=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32).flatten().tobytes(),
        raw=True
    )

    # Create operations that mix scalars and tensors
    nodes = [
        # Scalar + Tensor (broadcasting)
        helper.make_node('Add', ['input_tensor', 'input_scalar'], ['temp1'], name='add_tensor_scalar'),

        # Result * const_scalar (tensor * scalar)
        helper.make_node('Mul', ['temp1', 'const_scalar'], ['temp2'], name='mul_scalar'),

        # Add tensor constant (tensor + tensor)
        helper.make_node('Add', ['temp2', 'const_tensor'], ['output'], name='add_tensor'),
    ]

    # Create the graph
    graph = helper.make_graph(
        nodes,
        'type_mismatch_model',
        [input_tensor, input_scalar],
        [output],
        initializer=[const_scalar, const_tensor]
    )

    # Create the model
    model = helper.make_model(graph, producer_name="onnx-ir-test", opset_imports=[helper.make_opsetid("", 16)])

    # Check the model
    onnx.checker.check_model(model)

    return model


def main():
    """Generate and save the ONNX model."""
    model = create_type_mismatch_model()

    # Save the model
    output_path = '../fixtures/type_validation.onnx'
    onnx.save(model, output_path)
    print(f"Model saved to {output_path}")

    print(f"\nModel info:")
    print(f"  Tests scalar vs tensor type handling")
    print(f"  Mixed scalar (rank-0) and tensor (rank-2) operations")
    print(f"  Type inference with broadcasting")


if __name__ == '__main__':
    main()
