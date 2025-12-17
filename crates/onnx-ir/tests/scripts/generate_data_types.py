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
- Float types: F32, F64, F16, BF16
- Signed integer types: I8, I16, I32, I64
- Unsigned integer types: U8, U16, U32, U64
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
    # Float types
    input_f32 = helper.make_tensor_value_info('input_f32', TensorProto.FLOAT, [2, 3])
    input_f64 = helper.make_tensor_value_info('input_f64', TensorProto.DOUBLE, [2, 3])
    input_f16 = helper.make_tensor_value_info('input_f16', TensorProto.FLOAT16, [2, 3])
    input_bf16 = helper.make_tensor_value_info('input_bf16', TensorProto.BFLOAT16, [2, 3])

    # Signed integer types
    input_i8 = helper.make_tensor_value_info('input_i8', TensorProto.INT8, [2, 3])
    input_i16 = helper.make_tensor_value_info('input_i16', TensorProto.INT16, [2, 3])
    input_i32 = helper.make_tensor_value_info('input_i32', TensorProto.INT32, [2, 3])
    input_i64 = helper.make_tensor_value_info('input_i64', TensorProto.INT64, [2, 3])

    # Unsigned integer types
    input_u8 = helper.make_tensor_value_info('input_u8', TensorProto.UINT8, [2, 3])
    input_u16 = helper.make_tensor_value_info('input_u16', TensorProto.UINT16, [2, 3])
    input_u32 = helper.make_tensor_value_info('input_u32', TensorProto.UINT32, [2, 3])
    input_u64 = helper.make_tensor_value_info('input_u64', TensorProto.UINT64, [2, 3])

    # Outputs
    output_f32 = helper.make_tensor_value_info('output_f32', TensorProto.FLOAT, [2, 3])
    output_f64 = helper.make_tensor_value_info('output_f64', TensorProto.DOUBLE, [2, 3])
    output_f16 = helper.make_tensor_value_info('output_f16', TensorProto.FLOAT16, [2, 3])
    output_bf16 = helper.make_tensor_value_info('output_bf16', TensorProto.BFLOAT16, [2, 3])
    output_i8 = helper.make_tensor_value_info('output_i8', TensorProto.INT8, [2, 3])
    output_i16 = helper.make_tensor_value_info('output_i16', TensorProto.INT16, [2, 3])
    output_i32 = helper.make_tensor_value_info('output_i32', TensorProto.INT32, [2, 3])
    output_i64 = helper.make_tensor_value_info('output_i64', TensorProto.INT64, [2, 3])
    output_u8 = helper.make_tensor_value_info('output_u8', TensorProto.UINT8, [2, 3])
    output_u16 = helper.make_tensor_value_info('output_u16', TensorProto.UINT16, [2, 3])
    output_u32 = helper.make_tensor_value_info('output_u32', TensorProto.UINT32, [2, 3])
    output_u64 = helper.make_tensor_value_info('output_u64', TensorProto.UINT64, [2, 3])
    output_bool = helper.make_tensor_value_info('output_bool', TensorProto.BOOL, [2, 3])

    # Initializers with different types (using raw bytes for proper encoding)
    const_f32 = helper.make_tensor(
        name='const_f32',
        data_type=TensorProto.FLOAT,
        dims=[2, 3],
        vals=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32).flatten().tobytes(),
        raw=True
    )

    const_f64 = helper.make_tensor(
        name='const_f64',
        data_type=TensorProto.DOUBLE,
        dims=[2, 3],
        vals=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64).flatten().tobytes(),
        raw=True
    )

    const_f16 = helper.make_tensor(
        name='const_f16',
        data_type=TensorProto.FLOAT16,
        dims=[2, 3],
        vals=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float16).flatten().tobytes(),
        raw=True
    )

    const_bf16 = helper.make_tensor(
        name='const_bf16',
        data_type=TensorProto.BFLOAT16,
        dims=[2, 3],
        # BF16 doesn't have native numpy support, so we use uint16 view of float32
        vals=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32).view(np.uint16)[1::2].tobytes(),
        raw=True
    )

    const_i8 = helper.make_tensor(
        name='const_i8',
        data_type=TensorProto.INT8,
        dims=[2, 3],
        vals=np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int8).flatten().tobytes(),
        raw=True
    )

    const_i16 = helper.make_tensor(
        name='const_i16',
        data_type=TensorProto.INT16,
        dims=[2, 3],
        vals=np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int16).flatten().tobytes(),
        raw=True
    )

    const_i32 = helper.make_tensor(
        name='const_i32',
        data_type=TensorProto.INT32,
        dims=[2, 3],
        vals=np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32).flatten().tobytes(),
        raw=True
    )

    const_i64 = helper.make_tensor(
        name='const_i64',
        data_type=TensorProto.INT64,
        dims=[2, 3],
        vals=np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64).flatten().tobytes(),
        raw=True
    )

    const_u8 = helper.make_tensor(
        name='const_u8',
        data_type=TensorProto.UINT8,
        dims=[2, 3],
        vals=np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8).flatten().tobytes(),
        raw=True
    )

    const_u16 = helper.make_tensor(
        name='const_u16',
        data_type=TensorProto.UINT16,
        dims=[2, 3],
        vals=np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint16).flatten().tobytes(),
        raw=True
    )

    const_u32 = helper.make_tensor(
        name='const_u32',
        data_type=TensorProto.UINT32,
        dims=[2, 3],
        vals=np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint32).flatten().tobytes(),
        raw=True
    )

    const_u64 = helper.make_tensor(
        name='const_u64',
        data_type=TensorProto.UINT64,
        dims=[2, 3],
        vals=np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint64).flatten().tobytes(),
        raw=True
    )

    # Create nodes testing each type
    # Using Add for types that support it, Identity for others
    nodes = [
        # Float types
        helper.make_node('Add', ['input_f32', 'const_f32'], ['output_f32'], name='add_f32'),
        helper.make_node('Add', ['input_f64', 'const_f64'], ['output_f64'], name='add_f64'),
        helper.make_node('Add', ['input_f16', 'const_f16'], ['output_f16'], name='add_f16'),
        helper.make_node('Add', ['input_bf16', 'const_bf16'], ['output_bf16'], name='add_bf16'),

        # Signed integer types
        helper.make_node('Add', ['input_i8', 'const_i8'], ['output_i8'], name='add_i8'),
        helper.make_node('Add', ['input_i16', 'const_i16'], ['output_i16'], name='add_i16'),
        helper.make_node('Add', ['input_i32', 'const_i32'], ['output_i32'], name='add_i32'),
        helper.make_node('Add', ['input_i64', 'const_i64'], ['output_i64'], name='add_i64'),

        # Unsigned integer types
        helper.make_node('Add', ['input_u8', 'const_u8'], ['output_u8'], name='add_u8'),
        helper.make_node('Add', ['input_u16', 'const_u16'], ['output_u16'], name='add_u16'),
        helper.make_node('Add', ['input_u32', 'const_u32'], ['output_u32'], name='add_u32'),
        helper.make_node('Add', ['input_u64', 'const_u64'], ['output_u64'], name='add_u64'),

        # Bool: Greater comparison (produces bool output)
        helper.make_node('Greater', ['input_f32', 'const_f32'], ['output_bool'], name='greater'),
    ]

    # Create the graph
    graph = helper.make_graph(
        nodes,
        'data_types_model',
        [input_f32, input_f64, input_f16, input_bf16,
         input_i8, input_i16, input_i32, input_i64,
         input_u8, input_u16, input_u32, input_u64],
        [output_f32, output_f64, output_f16, output_bf16,
         output_i8, output_i16, output_i32, output_i64,
         output_u8, output_u16, output_u32, output_u64,
         output_bool],
        initializer=[const_f32, const_f64, const_f16, const_bf16,
                     const_i8, const_i16, const_i32, const_i64,
                     const_u8, const_u16, const_u32, const_u64]
    )

    # Create the model
    model = helper.make_model(graph, producer_name="onnx-ir-test", opset_imports=[helper.make_opsetid("", 21)])

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
    print(f"  Inputs: {len(model.graph.input)}")
    for inp in model.graph.input:
        dtype_name = TensorProto.DataType.Name(inp.type.tensor_type.elem_type)
        print(f"    - {inp.name}: {dtype_name}")
    print(f"  Outputs: {len(model.graph.output)}")
    for out in model.graph.output:
        dtype_name = TensorProto.DataType.Name(out.type.tensor_type.elem_type)
        print(f"    - {out.name}: {dtype_name}")
    print(f"  Initializers: {len(model.graph.initializer)}")
    for init in model.graph.initializer:
        dtype_name = TensorProto.DataType.Name(init.data_type)
        print(f"    - {init.name}: {dtype_name}")
    print(f"  Nodes: {len(model.graph.node)}")
    for node in model.graph.node:
        print(f"    - {node.op_type} ({node.name})")


if __name__ == '__main__':
    main()
