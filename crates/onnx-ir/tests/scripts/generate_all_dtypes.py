#!/usr/bin/env -S uv run
# /// script
# dependencies = [
#   "onnx>=1.15.0",
#   "numpy>=1.24.0",
# ]
# ///

"""
Generate ONNX model that validates ALL data type conversions.

Creates initializers of different types and uses them in Identity operations
so they become Constant nodes in the IR and don't get removed.
"""

import onnx
from onnx import helper, TensorProto, numpy_helper
import numpy as np


def create_all_dtypes_model():
    """Create model with all data types using proper Constant nodes."""

    nodes = []
    outputs = []

    # ==========================================================================
    # F32 - non-empty, empty, scalar
    # ==========================================================================
    f32_data = np.array([1.5, -2.5, 3.14159, 0.0, float('inf'), float('-inf'), 2.71828], dtype=np.float32)
    f32_tensor = numpy_helper.from_array(f32_data, name='f32_data')
    nodes.append(helper.make_node('Constant', [], ['f32_out'], value=f32_tensor, name='f32_const'))
    outputs.append(helper.make_tensor_value_info('f32_out', TensorProto.FLOAT, [7]))

    f32_empty_tensor = numpy_helper.from_array(np.array([], dtype=np.float32), name='f32_empty_data')
    f32_empty_tensor.dims[:] = [0]
    nodes.append(helper.make_node('Constant', [], ['f32_empty_out'], value=f32_empty_tensor, name='f32_empty_const'))
    outputs.append(helper.make_tensor_value_info('f32_empty_out', TensorProto.FLOAT, [0]))

    f32_scalar_tensor = numpy_helper.from_array(np.array(42.0, dtype=np.float32), name='f32_scalar_data')
    nodes.append(helper.make_node('Constant', [], ['f32_scalar_out'], value=f32_scalar_tensor, name='f32_scalar_const'))
    outputs.append(helper.make_tensor_value_info('f32_scalar_out', TensorProto.FLOAT, []))

    # ==========================================================================
    # F64
    # ==========================================================================
    f64_tensor = numpy_helper.from_array(np.array([1.5e100, -2.5e-100, 3.141592653589793], dtype=np.float64), name='f64_data')
    nodes.append(helper.make_node('Constant', [], ['f64_out'], value=f64_tensor, name='f64_const'))
    outputs.append(helper.make_tensor_value_info('f64_out', TensorProto.DOUBLE, [3]))

    f64_empty_tensor = numpy_helper.from_array(np.array([], dtype=np.float64), name='f64_empty_data')
    f64_empty_tensor.dims[:] = [0]
    nodes.append(helper.make_node('Constant', [], ['f64_empty_out'], value=f64_empty_tensor, name='f64_empty_const'))
    outputs.append(helper.make_tensor_value_info('f64_empty_out', TensorProto.DOUBLE, [0]))

    # ==========================================================================
    # I32
    # ==========================================================================
    i32_tensor = numpy_helper.from_array(np.array([2147483647, -2147483648, 0, 1, -1], dtype=np.int32), name='i32_data')
    nodes.append(helper.make_node('Constant', [], ['i32_out'], value=i32_tensor, name='i32_const'))
    outputs.append(helper.make_tensor_value_info('i32_out', TensorProto.INT32, [5]))

    i32_empty_tensor = numpy_helper.from_array(np.array([], dtype=np.int32), name='i32_empty_data')
    i32_empty_tensor.dims[:] = [0]
    nodes.append(helper.make_node('Constant', [], ['i32_empty_out'], value=i32_empty_tensor, name='i32_empty_const'))
    outputs.append(helper.make_tensor_value_info('i32_empty_out', TensorProto.INT32, [0]))

    # ==========================================================================
    # I64
    # ==========================================================================
    i64_tensor = numpy_helper.from_array(np.array([9223372036854775807, -9223372036854775808, 0], dtype=np.int64), name='i64_data')
    nodes.append(helper.make_node('Constant', [], ['i64_out'], value=i64_tensor, name='i64_const'))
    outputs.append(helper.make_tensor_value_info('i64_out', TensorProto.INT64, [3]))

    i64_empty_tensor = numpy_helper.from_array(np.array([], dtype=np.int64), name='i64_empty_data')
    i64_empty_tensor.dims[:] = [0]
    nodes.append(helper.make_node('Constant', [], ['i64_empty_out'], value=i64_empty_tensor, name='i64_empty_const'))
    outputs.append(helper.make_tensor_value_info('i64_empty_out', TensorProto.INT64, [0]))

    # ==========================================================================
    # Bool
    # ==========================================================================
    bool_tensor = numpy_helper.from_array(np.array([True, False, True, True, False], dtype=bool), name='bool_data')
    nodes.append(helper.make_node('Constant', [], ['bool_out'], value=bool_tensor, name='bool_const'))
    outputs.append(helper.make_tensor_value_info('bool_out', TensorProto.BOOL, [5]))

    bool_empty_tensor = numpy_helper.from_array(np.array([], dtype=bool), name='bool_empty_data')
    bool_empty_tensor.dims[:] = [0]
    nodes.append(helper.make_node('Constant', [], ['bool_empty_out'], value=bool_empty_tensor, name='bool_empty_const'))
    outputs.append(helper.make_tensor_value_info('bool_empty_out', TensorProto.BOOL, [0]))

    # ==========================================================================
    # U8
    # ==========================================================================
    u8_tensor = numpy_helper.from_array(np.array([0, 255, 128, 1, 254], dtype=np.uint8), name='u8_data')
    nodes.append(helper.make_node('Constant', [], ['u8_out'], value=u8_tensor, name='u8_const'))
    outputs.append(helper.make_tensor_value_info('u8_out', TensorProto.UINT8, [5]))

    u8_empty_tensor = numpy_helper.from_array(np.array([], dtype=np.uint8), name='u8_empty_data')
    u8_empty_tensor.dims[:] = [0]
    nodes.append(helper.make_node('Constant', [], ['u8_empty_out'], value=u8_empty_tensor, name='u8_empty_const'))
    outputs.append(helper.make_tensor_value_info('u8_empty_out', TensorProto.UINT8, [0]))

    # ==========================================================================
    # I8
    # ==========================================================================
    i8_tensor = numpy_helper.from_array(np.array([127, -128, 0, 1, -1], dtype=np.int8), name='i8_data')
    nodes.append(helper.make_node('Constant', [], ['i8_out'], value=i8_tensor, name='i8_const'))
    outputs.append(helper.make_tensor_value_info('i8_out', TensorProto.INT8, [5]))

    i8_empty_tensor = numpy_helper.from_array(np.array([], dtype=np.int8), name='i8_empty_data')
    i8_empty_tensor.dims[:] = [0]
    nodes.append(helper.make_node('Constant', [], ['i8_empty_out'], value=i8_empty_tensor, name='i8_empty_const'))
    outputs.append(helper.make_tensor_value_info('i8_empty_out', TensorProto.INT8, [0]))

    # ==========================================================================
    # F16
    # ==========================================================================
    f16_tensor = numpy_helper.from_array(np.array([1.0, -1.0, 0.5, 65504.0], dtype=np.float16), name='f16_data')
    nodes.append(helper.make_node('Constant', [], ['f16_out'], value=f16_tensor, name='f16_const'))
    outputs.append(helper.make_tensor_value_info('f16_out', TensorProto.FLOAT16, [4]))

    f16_empty_tensor = numpy_helper.from_array(np.array([], dtype=np.float16), name='f16_empty_data')
    f16_empty_tensor.dims[:] = [0]
    nodes.append(helper.make_node('Constant', [], ['f16_empty_out'], value=f16_empty_tensor, name='f16_empty_const'))
    outputs.append(helper.make_tensor_value_info('f16_empty_out', TensorProto.FLOAT16, [0]))

    # ==========================================================================
    # U16
    # ==========================================================================
    u16_tensor = numpy_helper.from_array(np.array([0, 65535, 32768, 1], dtype=np.uint16), name='u16_data')
    nodes.append(helper.make_node('Constant', [], ['u16_out'], value=u16_tensor, name='u16_const'))
    outputs.append(helper.make_tensor_value_info('u16_out', TensorProto.UINT16, [4]))

    u16_empty_tensor = numpy_helper.from_array(np.array([], dtype=np.uint16), name='u16_empty_data')
    u16_empty_tensor.dims[:] = [0]
    nodes.append(helper.make_node('Constant', [], ['u16_empty_out'], value=u16_empty_tensor, name='u16_empty_const'))
    outputs.append(helper.make_tensor_value_info('u16_empty_out', TensorProto.UINT16, [0]))

    # ==========================================================================
    # Multi-dimensional tensors to test shape handling
    # ==========================================================================

    # 2D F32 tensor: [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    f32_2d = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    f32_2d_tensor = numpy_helper.from_array(f32_2d, name='f32_2d_data')
    nodes.append(helper.make_node('Constant', [], ['f32_2d_out'], value=f32_2d_tensor, name='f32_2d_const'))
    outputs.append(helper.make_tensor_value_info('f32_2d_out', TensorProto.FLOAT, [2, 3]))

    # 3D I32 tensor: [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
    i32_3d = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)
    i32_3d_tensor = numpy_helper.from_array(i32_3d, name='i32_3d_data')
    nodes.append(helper.make_node('Constant', [], ['i32_3d_out'], value=i32_3d_tensor, name='i32_3d_const'))
    outputs.append(helper.make_tensor_value_info('i32_3d_out', TensorProto.INT32, [2, 2, 2]))

    # 2D Bool tensor: [[True, False], [False, True]]
    bool_2d = np.array([[True, False], [False, True]], dtype=bool)
    bool_2d_tensor = numpy_helper.from_array(bool_2d, name='bool_2d_data')
    nodes.append(helper.make_node('Constant', [], ['bool_2d_out'], value=bool_2d_tensor, name='bool_2d_const'))
    outputs.append(helper.make_tensor_value_info('bool_2d_out', TensorProto.BOOL, [2, 2]))

    # Create the graph with NO inputs and NO initializers (all data in Constant nodes)
    graph = helper.make_graph(
        nodes,
        'all_dtypes_model',
        [],  # No inputs
        outputs,
    )

    # Create the model
    model = helper.make_model(
        graph,
        producer_name="onnx-ir-test",
        opset_imports=[helper.make_opsetid("", 16)]
    )

    # Check the model
    onnx.checker.check_model(model)

    return model


def main():
    """Generate and save the ONNX model."""
    model = create_all_dtypes_model()

    # Save the model
    output_path = '../fixtures/all_dtypes.onnx'
    onnx.save(model, output_path)
    print(f"Model saved to {output_path}")

    print(f"\nModel info:")
    print(f"  Total Constant nodes: {len(model.graph.node)}")
    print(f"  Total initializers: {len(model.graph.initializer)} (should be 0)")
    print(f"  Data types tested:")
    print(f"    - FLOAT (f32): 1D [7], empty [0], scalar [], 2D [2,3]")
    print(f"    - DOUBLE (f64): 1D [3], empty [0]")
    print(f"    - INT32 (i32): 1D [5], empty [0], 3D [2,2,2]")
    print(f"    - INT64 (i64): 1D [3], empty [0]")
    print(f"    - BOOL: 1D [5], empty [0], 2D [2,2]")
    print(f"    - UINT8 (u8): 1D [5], empty [0]")
    print(f"    - INT8 (i8): 1D [5], empty [0]")
    print(f"    - FLOAT16 (f16): 1D [4], empty [0]")
    print(f"    - UINT16 (u16): 1D [4], empty [0]")
    print(f"  Multi-dimensional shapes:")
    print(f"    - F32: [2, 3] (2D)")
    print(f"    - I32: [2, 2, 2] (3D)")
    print(f"    - Bool: [2, 2] (2D)")
    print(f"  Each Constant node has 0 inputs (no 'value' attribute)")
    print(f"  After IR conversion, 'value' attribute becomes first input")
    print(f"  Tests correctness of tensor data conversion for all types and shapes")


if __name__ == '__main__':
    main()
