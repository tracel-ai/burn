#!/usr/bin/env -S uv run
# /// script
# dependencies = [
#   "onnx>=1.15.0",
#   "numpy>=1.24.0",
# ]
# ///

"""
Generate ONNX model testing different data types.

Tests:
- Float types: F32, F64
- Integer types: I32, I64
- Boolean type
- Mixed types in single graph
- Type preservation through pipeline
"""

import onnx
from onnx import helper, TensorProto
import numpy as np


def create_data_types_model():
    """Create model with various data types."""

    # Inputs with different types
    input_f32 = helper.make_tensor_value_info('input_f32', TensorProto.FLOAT, [2, 3])
    input_f64 = helper.make_tensor_value_info('input_f64', TensorProto.DOUBLE, [2, 3])
    input_i32 = helper.make_tensor_value_info('input_i32', TensorProto.INT32, [2, 3])
    input_i64 = helper.make_tensor_value_info('input_i64', TensorProto.INT64, [2, 3])

    # Outputs
    output_f32 = helper.make_tensor_value_info('output_f32', TensorProto.FLOAT, [2, 3])
    output_f64 = helper.make_tensor_value_info('output_f64', TensorProto.DOUBLE, [2, 3])
    output_i32 = helper.make_tensor_value_info('output_i32', TensorProto.INT32, [2, 3])
    output_i64 = helper.make_tensor_value_info('output_i64', TensorProto.INT64, [2, 3])
    output_bool = helper.make_tensor_value_info('output_bool', TensorProto.BOOL, [2, 3])

    # Initializers with different types
    const_f32 = helper.make_tensor(
        name='const_f32',
        data_type=TensorProto.FLOAT,
        dims=[2, 3],
        vals=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32).flatten().tobytes(),
        raw=True
    )

    const_i64 = helper.make_tensor(
        name='const_i64',
        data_type=TensorProto.INT64,
        dims=[2, 3],
        vals=np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64).flatten().tobytes(),
        raw=True
    )

    # Create nodes testing each type
    nodes = [
        # F32: Add operation
        helper.make_node('Add', ['input_f32', 'const_f32'], ['output_f32'], name='add_f32'),

        # F64: Abs operation
        helper.make_node('Abs', ['input_f64'], ['output_f64'], name='abs_f64'),

        # I32: Neg operation
        helper.make_node('Neg', ['input_i32'], ['output_i32'], name='neg_i32'),

        # I64: Add operation
        helper.make_node('Add', ['input_i64', 'const_i64'], ['output_i64'], name='add_i64'),

        # Bool: Greater comparison (produces bool output)
        helper.make_node('Greater', ['input_f32', 'const_f32'], ['output_bool'], name='greater'),
    ]

    # Create the graph
    graph = helper.make_graph(
        nodes,
        'data_types_model',
        [input_f32, input_f64, input_i32, input_i64],
        [output_f32, output_f64, output_i32, output_i64, output_bool],
        initializer=[const_f32, const_i64]
    )

    # Create the model
    model = helper.make_model(graph, producer_name="onnx-ir-test", opset_imports=[helper.make_opsetid("", 16)])

    # Check the model
    onnx.checker.check_model(model)

    return model


def main():
    """Generate and save the ONNX model."""
    model = create_data_types_model()

    # Save the model
    output_path = '../fixtures/data_types.onnx'
    onnx.save(model, output_path)
    print(f"Model saved to {output_path}")

    # Print model info
    print(f"\nModel info:")
    print(f"  Opset version: {model.opset_import[0].version}")
    print(f"  Inputs: {[(inp.name, inp.type.tensor_type.elem_type) for inp in model.graph.input]}")
    print(f"  Outputs: {[(out.name, out.type.tensor_type.elem_type) for out in model.graph.output]}")
    print(f"  Nodes: {len(model.graph.node)}")
    for node in model.graph.node:
        print(f"    - {node.op_type} ({node.name})")


if __name__ == '__main__':
    main()
