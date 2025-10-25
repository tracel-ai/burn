#!/usr/bin/env python3

# Used to generate multiple EyeLike ONNX test models

import numpy as np
import onnx
from onnx import helper, TensorProto


def build_eye_like_model():
    """Build basic EyeLike model (k=0, default dtype)"""
    input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, ['H', 'W'])
    output_tensor = helper.make_tensor_value_info('output', TensorProto.FLOAT, ['H', 'W'])

    eye_like_node = helper.make_node(
        'EyeLike',
        inputs=['input'],
        outputs=['output'],
        name='EyeLike_0'
    )

    graph = helper.make_graph(
        [eye_like_node],
        'EyeLikeModel',
        [input_tensor],
        [output_tensor]
    )

    return helper.make_model(
        graph,
        producer_name='eye_like_test',
        opset_imports=[helper.make_opsetid("", 16)]
    )


def build_eye_like_k_minus1_model():
    """Build EyeLike model with k=-1 (lower diagonal)"""
    input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, [4, 4])
    output_tensor = helper.make_tensor_value_info('output', TensorProto.FLOAT, [4, 4])

    eye_like_node = helper.make_node(
        'EyeLike',
        inputs=['input'],
        outputs=['output'],
        name='EyeLike_k_minus1',
        k=-1
    )

    graph = helper.make_graph(
        [eye_like_node],
        'EyeLikeKMinus1Model',
        [input_tensor],
        [output_tensor]
    )

    return helper.make_model(
        graph,
        producer_name='eye_like_k_minus1_test',
        opset_imports=[helper.make_opsetid("", 16)]
    )


def build_eye_like_float64_model():
    """Build EyeLike model with Float64 dtype"""
    input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, [3, 3])
    output_tensor = helper.make_tensor_value_info('output', TensorProto.DOUBLE, [3, 3])

    eye_like_node = helper.make_node(
        'EyeLike',
        inputs=['input'],
        outputs=['output'],
        name='EyeLike_float64',
        dtype=TensorProto.DOUBLE
    )

    graph = helper.make_graph(
        [eye_like_node],
        'EyeLikeFloat64Model',
        [input_tensor],
        [output_tensor]
    )

    return helper.make_model(
        graph,
        producer_name='eye_like_float64_test',
        opset_imports=[helper.make_opsetid("", 16)]
    )


def build_eye_like_int32_model():
    """Build EyeLike model with Int32 dtype"""
    input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, [3, 3])
    output_tensor = helper.make_tensor_value_info('output', TensorProto.INT32, [3, 3])

    eye_like_node = helper.make_node(
        'EyeLike',
        inputs=['input'],
        outputs=['output'],
        name='EyeLike_int32',
        dtype=TensorProto.INT32
    )

    graph = helper.make_graph(
        [eye_like_node],
        'EyeLikeInt32Model',
        [input_tensor],
        [output_tensor]
    )

    return helper.make_model(
        graph,
        producer_name='eye_like_int32_test',
        opset_imports=[helper.make_opsetid("", 16)]
    )


def build_eye_like_bool_model():
    """Build EyeLike model with Bool dtype"""
    input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, [3, 3])
    output_tensor = helper.make_tensor_value_info('output', TensorProto.BOOL, [3, 3])

    eye_like_node = helper.make_node(
        'EyeLike',
        inputs=['input'],
        outputs=['output'],
        name='EyeLike_bool',
        dtype=TensorProto.BOOL
    )

    graph = helper.make_graph(
        [eye_like_node],
        'EyeLikeBoolModel',
        [input_tensor],
        [output_tensor]
    )

    return helper.make_model(
        graph,
        producer_name='eye_like_bool_test',
        opset_imports=[helper.make_opsetid("", 16)]
    )


def build_eye_like_large_k_model():
    """Build EyeLike model with k=5 (way beyond matrix size for edge case)"""
    input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, [3, 3])
    output_tensor = helper.make_tensor_value_info('output', TensorProto.FLOAT, [3, 3])

    eye_like_node = helper.make_node(
        'EyeLike',
        inputs=['input'],
        outputs=['output'],
        name='EyeLike_large_k',
        k=5  # Way beyond 3x3 matrix bounds
    )

    graph = helper.make_graph(
        [eye_like_node],
        'EyeLikeLargeKModel',
        [input_tensor],
        [output_tensor]
    )

    return helper.make_model(
        graph,
        producer_name='eye_like_large_k_test',
        opset_imports=[helper.make_opsetid("", 16)]
    )


def build_eye_like_small_matrix_model():
    """Build EyeLike model with 1x1 matrix (smallest possible)"""
    input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 1])
    output_tensor = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 1])

    eye_like_node = helper.make_node(
        'EyeLike',
        inputs=['input'],
        outputs=['output'],
        name='EyeLike_1x1'
    )

    graph = helper.make_graph(
        [eye_like_node],
        'EyeLike1x1Model',
        [input_tensor],
        [output_tensor]
    )

    return helper.make_model(
        graph,
        producer_name='eye_like_1x1_test',
        opset_imports=[helper.make_opsetid("", 16)]
    )


def build_eye_like_wide_matrix_model():
    """Build EyeLike model with very wide matrix (1x8)"""
    input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 8])
    output_tensor = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 8])

    eye_like_node = helper.make_node(
        'EyeLike',
        inputs=['input'],
        outputs=['output'],
        name='EyeLike_wide',
        k=0
    )

    graph = helper.make_graph(
        [eye_like_node],
        'EyeLikeWideModel',
        [input_tensor],
        [output_tensor]
    )

    return helper.make_model(
        graph,
        producer_name='eye_like_wide_test',
        opset_imports=[helper.make_opsetid("", 16)]
    )


def build_eye_like_negative_large_k_model():
    """Build EyeLike model with k=-5 (way beyond matrix size negative)"""
    input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, [3, 3])
    output_tensor = helper.make_tensor_value_info('output', TensorProto.FLOAT, [3, 3])

    eye_like_node = helper.make_node(
        'EyeLike',
        inputs=['input'],
        outputs=['output'],
        name='EyeLike_neg_large_k',
        k=-5  # Way beyond 3x3 matrix bounds (negative)
    )

    graph = helper.make_graph(
        [eye_like_node],
        'EyeLikeNegLargeKModel',
        [input_tensor],
        [output_tensor]
    )

    return helper.make_model(
        graph,
        producer_name='eye_like_neg_large_k_test',
        opset_imports=[helper.make_opsetid("", 16)]
    )


def main():
    # Generate all test models including edge cases
    models = [
        ('eye_like.onnx', build_eye_like_model()),
        ('eye_like_k_minus1.onnx', build_eye_like_k_minus1_model()),
        ('eye_like_float64.onnx', build_eye_like_float64_model()),
        ('eye_like_int32.onnx', build_eye_like_int32_model()),
        ('eye_like_bool.onnx', build_eye_like_bool_model()),
        # Edge case models
        ('eye_like_large_k.onnx', build_eye_like_large_k_model()),
        ('eye_like_1x1.onnx', build_eye_like_small_matrix_model()),
        ('eye_like_wide.onnx', build_eye_like_wide_matrix_model()),
        ('eye_like_neg_large_k.onnx', build_eye_like_negative_large_k_model()),
    ]

    for filename, model in models:
        onnx.save(model, filename)
        print(f"Generated {filename}")

    print("\nEyeLike operator test cases:")
    print("- eye_like.onnx: Basic k=0 (main diagonal)")
    print("- eye_like_k1.onnx: k=1 (upper diagonal)")
    print("- eye_like_k_minus1.onnx: k=-1 (lower diagonal)")
    print("- eye_like_int.onnx: Int64 dtype")
    print("- eye_like_int32.onnx: Int32 dtype")
    print("- eye_like_float64.onnx: Float64 dtype")
    print("- eye_like_bool.onnx: Bool dtype")
    print("\nEdge case models:")
    print("- eye_like_large_k.onnx: k=5 (beyond matrix bounds)")
    print("- eye_like_neg_large_k.onnx: k=-5 (negative beyond bounds)")
    print("- eye_like_1x1.onnx: Smallest possible matrix")
    print("- eye_like_wide.onnx: Very wide matrix (1x8)")


if __name__ == '__main__':
    main()