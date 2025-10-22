#!/usr/bin/env -S uv run
# /// script
# dependencies = [
#   "onnx>=1.15.0",
#   "numpy>=1.24.0",
# ]
# ///

"""
Generate ONNX model testing different argument types.

Tests:
- Scalar arguments (rank-0 tensors)
- Shape arguments (from Shape operator)
- Tensor arguments with various ranks (1D, 2D, 3D, 4D)
- Type inference with mixed argument types
- Shape preference propagation
"""

import onnx
from onnx import helper, TensorProto
import numpy as np


def create_arg_types_model():
    """Create model with Scalar, Shape, and Tensor argument types."""

    # Inputs with different ranks
    input_4d = helper.make_tensor_value_info('input_4d', TensorProto.FLOAT, [1, 3, 4, 4])  # 4D tensor
    input_2d = helper.make_tensor_value_info('input_2d', TensorProto.FLOAT, [3, 5])        # 2D tensor
    input_1d = helper.make_tensor_value_info('input_1d', TensorProto.FLOAT, [10])          # 1D tensor

    # Outputs
    output_shape = helper.make_tensor_value_info('output_shape', TensorProto.INT64, [4])    # Shape output
    output_scalar = helper.make_tensor_value_info('output_scalar', TensorProto.FLOAT, [])  # Scalar output
    output_reshaped = helper.make_tensor_value_info('output_reshaped', TensorProto.FLOAT, [1, 3, 16]) # Reshaped tensor

    # Initializers
    # Target shape for Reshape (will be lifted to Static)
    target_shape = helper.make_tensor(
        name='target_shape',
        data_type=TensorProto.INT64,
        dims=[3],
        vals=np.array([1, 3, 16], dtype=np.int64).tobytes(),
        raw=True
    )

    # Scalar constant for ReduceSum
    scalar_const = helper.make_tensor(
        name='scalar_value',
        data_type=TensorProto.FLOAT,
        dims=[],  # Empty dims = scalar
        vals=np.array([2.0], dtype=np.float32).tobytes(),
        raw=True
    )

    # Reduction axes
    axes = helper.make_tensor(
        name='axes',
        data_type=TensorProto.INT64,
        dims=[1],
        vals=np.array([0], dtype=np.int64).tobytes(),
        raw=True
    )

    # Create nodes
    nodes = [
        # Shape operator: Tensor → Shape (1D int64 vector)
        helper.make_node('Shape', ['input_4d'], ['output_shape'], name='shape'),

        # ReduceSum: Tensor → Scalar (by reducing all dimensions)
        helper.make_node(
            'ReduceSum',
            ['input_1d', 'axes'],
            ['sum_result'],
            name='reduce_sum',
            keepdims=0  # No keepdims = scalar output
        ),

        # Mul: Scalar × Scalar → Scalar
        helper.make_node('Mul', ['sum_result', 'scalar_value'], ['output_scalar'], name='mul_scalar'),

        # Reshape: Tensor + Shape → Tensor with new shape
        # This tests Shape argument type and constant lifting
        helper.make_node('Reshape', ['input_4d', 'target_shape'], ['output_reshaped'], name='reshape'),
    ]

    # Create the graph
    graph = helper.make_graph(
        nodes,
        'arg_types_model',
        [input_4d, input_2d, input_1d],
        [output_shape, output_scalar, output_reshaped],
        initializer=[target_shape, scalar_const, axes]
    )

    # Create the model
    model = helper.make_model(graph, producer_name="onnx-ir-test", opset_imports=[helper.make_opsetid("", 16)])

    # Check the model
    onnx.checker.check_model(model)

    return model


def main():
    """Generate and save the ONNX model."""
    model = create_arg_types_model()

    # Save the model
    output_path = '../fixtures/arg_types.onnx'
    onnx.save(model, output_path)
    print(f"Model saved to {output_path}")

    # Print model info
    print(f"\nModel info:")
    print(f"  Opset version: {model.opset_import[0].version}")
    print(f"  Inputs:")
    for inp in model.graph.input:
        shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]
        print(f"    - {inp.name}: shape={shape}")
    print(f"  Outputs:")
    for out in model.graph.output:
        shape = [d.dim_value for d in out.type.tensor_type.shape.dim]
        print(f"    - {out.name}: shape={shape}")
    print(f"  Nodes: {len(model.graph.node)}")
    for node in model.graph.node:
        print(f"    - {node.op_type} ({node.name}): {list(node.input)} → {list(node.output)}")


if __name__ == '__main__':
    main()
