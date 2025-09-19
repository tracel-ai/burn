#!/usr/bin/env python3

# Used to generate multiple NonZero ONNX test models

import numpy as np
import onnx
from onnx import helper, TensorProto

def build_nonzero_float32_model():
    """Build NonZero model for Float32 tensor"""
    input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, [3, 4])
    output_tensor = helper.make_tensor_value_info('output', TensorProto.INT64, [None, 2])

    nonzero_node = helper.make_node(
        'NonZero',
        inputs=['input'],
        outputs=['output'],
        name='NonZero_float32'
    )

    graph = helper.make_graph(
        [nonzero_node],
        'NonZeroFloat32Model',
        [input_tensor],
        [output_tensor]
    )

    return helper.make_model(
        graph,
        producer_name='nonzero_float32_test',
        opset_imports=[helper.make_opsetid("", 18)]
    )

def build_nonzero_int64_model():
    """Build NonZero model for Int64 tensor"""
    input_tensor = helper.make_tensor_value_info('input', TensorProto.INT64, [2, 3])
    output_tensor = helper.make_tensor_value_info('output', TensorProto.INT64, [None, 2])

    nonzero_node = helper.make_node(
        'NonZero',
        inputs=['input'],
        outputs=['output'],
        name='NonZero_int64'
    )

    graph = helper.make_graph(
        [nonzero_node],
        'NonZeroInt64Model',
        [input_tensor],
        [output_tensor]
    )

    return helper.make_model(
        graph,
        producer_name='nonzero_int64_test',
        opset_imports=[helper.make_opsetid("", 18)]
    )

def build_nonzero_bool_model():
    """Build NonZero model for Bool tensor"""
    input_tensor = helper.make_tensor_value_info('input', TensorProto.BOOL, [2, 2])
    output_tensor = helper.make_tensor_value_info('output', TensorProto.INT64, [None, 2])

    nonzero_node = helper.make_node(
        'NonZero',
        inputs=['input'],
        outputs=['output'],
        name='NonZero_bool'
    )

    graph = helper.make_graph(
        [nonzero_node],
        'NonZeroBoolModel',
        [input_tensor],
        [output_tensor]
    )

    return helper.make_model(
        graph,
        producer_name='nonzero_bool_test',
        opset_imports=[helper.make_opsetid("", 18)]
    )

def build_nonzero_1d_model():
    """Build NonZero model for 1D tensor"""
    input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, [6])
    output_tensor = helper.make_tensor_value_info('output', TensorProto.INT64, [None, 1])

    nonzero_node = helper.make_node(
        'NonZero',
        inputs=['input'],
        outputs=['output'],
        name='NonZero_1d'
    )

    graph = helper.make_graph(
        [nonzero_node],
        'NonZero1DModel',
        [input_tensor],
        [output_tensor]
    )

    return helper.make_model(
        graph,
        producer_name='nonzero_1d_test',
        opset_imports=[helper.make_opsetid("", 18)]
    )

def build_nonzero_3d_model():
    """Build NonZero model for 3D tensor"""
    input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, [2, 2, 3])
    output_tensor = helper.make_tensor_value_info('output', TensorProto.INT64, [None, 3])

    nonzero_node = helper.make_node(
        'NonZero',
        inputs=['input'],
        outputs=['output'],
        name='NonZero_3d'
    )

    graph = helper.make_graph(
        [nonzero_node],
        'NonZero3DModel',
        [input_tensor],
        [output_tensor]
    )

    return helper.make_model(
        graph,
        producer_name='nonzero_3d_test',
        opset_imports=[helper.make_opsetid("", 18)]
    )

def build_nonzero_empty_model():
    """Build NonZero model that should return empty output (all zeros)"""
    input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, [2, 3])
    output_tensor = helper.make_tensor_value_info('output', TensorProto.INT64, [None, 2])

    nonzero_node = helper.make_node(
        'NonZero',
        inputs=['input'],
        outputs=['output'],
        name='NonZero_empty'
    )

    graph = helper.make_graph(
        [nonzero_node],
        'NonZeroEmptyModel',
        [input_tensor],
        [output_tensor]
    )

    return helper.make_model(
        graph,
        producer_name='nonzero_empty_test',
        opset_imports=[helper.make_opsetid("", 18)]
    )

def main():
    # Generate all test models
    models = [
        ('nonzero_float32.onnx', build_nonzero_float32_model()),
        ('nonzero_int64.onnx', build_nonzero_int64_model()),
        ('nonzero_bool.onnx', build_nonzero_bool_model()),
        ('nonzero_1d.onnx', build_nonzero_1d_model()),
        ('nonzero_3d.onnx', build_nonzero_3d_model()),
        ('nonzero_empty.onnx', build_nonzero_empty_model()),
    ]

    for filename, model in models:
        onnx.save(model, filename)
        print(f"Generated {filename}")

    print("\nNonZero operator test cases:")
    print("- nonzero_float32.onnx: Float32 2D tensor (3x4)")
    print("- nonzero_int64.onnx: Int64 2D tensor (2x3)")
    print("- nonzero_bool.onnx: Bool 2D tensor (2x2)")
    print("- nonzero_1d.onnx: Float32 1D tensor (6,)")
    print("- nonzero_3d.onnx: Float32 3D tensor (2x2x3)")
    print("- nonzero_empty.onnx: Test case for all-zero tensor")

    print("\nExpected behavior:")
    print("- Output is always Int64 tensor with shape [num_nonzero_elements, input_rank]")
    print("- Each row contains the indices of one non-zero element")
    print("- Order follows C-style (row-major) traversal")

if __name__ == '__main__':
    main()