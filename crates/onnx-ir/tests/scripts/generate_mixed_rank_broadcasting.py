#!/usr/bin/env -S uv run
# /// script
# dependencies = [
#   "onnx>=1.15.0",
#   "numpy>=1.24.0",
# ]
# ///

"""
Generate ONNX model with mixed-rank broadcasting operations.

Tests:
- NumPy-style broadcasting with different ranks
- Scalar (0D) + tensor operations
- 1D + 4D tensor operations
- Type inference with rank differences
"""

import onnx
from onnx import helper, TensorProto
import numpy as np


def create_mixed_rank_model():
    """Create model with operations on tensors of different ranks."""

    # Inputs with different ranks
    input_4d = helper.make_tensor_value_info('input_4d', TensorProto.FLOAT, [1, 3, 4, 4])  # 4D
    input_2d = helper.make_tensor_value_info('input_2d', TensorProto.FLOAT, [1, 3])        # 2D
    input_scalar = helper.make_tensor_value_info('input_scalar', TensorProto.FLOAT, [])    # Scalar

    # Outputs
    output1 = helper.make_tensor_value_info('output_4d_scalar', TensorProto.FLOAT, [1, 3, 4, 4])
    output2 = helper.make_tensor_value_info('output_4d_2d', TensorProto.FLOAT, [1, 3, 4, 4])

    # Scalar constant
    scalar_const = helper.make_tensor(
        name='scalar_const',
        data_type=TensorProto.FLOAT,
        dims=[],
        vals=np.array([2.0], dtype=np.float32).tobytes(),
        raw=True
    )

    # 1D constant
    const_1d = helper.make_tensor(
        name='const_1d',
        data_type=TensorProto.FLOAT,
        dims=[3],
        vals=np.array([1.0, 2.0, 3.0], dtype=np.float32).tobytes(),
        raw=True
    )

    # Create nodes with mixed-rank operations
    nodes = [
        # 4D + scalar (0D + 4D broadcasting)
        helper.make_node('Add', ['input_4d', 'input_scalar'], ['temp1'], name='add_4d_scalar'),

        # temp1 * scalar_const (4D * 0D)
        helper.make_node('Mul', ['temp1', 'scalar_const'], ['output_4d_scalar'], name='mul_4d_scalar'),

        # 4D + 2D broadcasting (different ranks)
        helper.make_node('Add', ['input_4d', 'input_2d'], ['temp2'], name='add_4d_2d'),

        # temp2 + 1D constant (4D + 1D)
        helper.make_node('Add', ['temp2', 'const_1d'], ['output_4d_2d'], name='add_4d_1d'),
    ]

    # Create the graph
    graph = helper.make_graph(
        nodes,
        'mixed_rank_broadcasting_model',
        [input_4d, input_2d, input_scalar],
        [output1, output2],
        initializer=[scalar_const, const_1d]
    )

    # Create the model
    model = helper.make_model(graph, producer_name="onnx-ir-test", opset_imports=[helper.make_opsetid("", 16)])

    # Check the model
    onnx.checker.check_model(model)

    return model


def main():
    """Generate and save the ONNX model."""
    model = create_mixed_rank_model()

    # Save the model
    output_path = '../fixtures/mixed_rank_broadcasting.onnx'
    onnx.save(model, output_path)
    print(f"Model saved to {output_path}")

    print(f"\nModel info:")
    print(f"  Operations with mixed ranks:")
    print(f"    - 4D + 0D (scalar)")
    print(f"    - 4D + 2D")
    print(f"    - 4D + 1D")
    print(f"  Tests NumPy-style broadcasting")


if __name__ == '__main__':
    main()
