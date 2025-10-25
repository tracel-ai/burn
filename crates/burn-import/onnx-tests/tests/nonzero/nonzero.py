#!/usr/bin/env python3

# Used to generate multiple NonZero ONNX test models

import numpy as np
import onnx
from onnx import helper, TensorProto
from onnx.reference import ReferenceEvaluator

def build_nonzero_float32_model():
    """Build NonZero model for Float32 tensor"""
    input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, [3, 4])
    output_tensor = helper.make_tensor_value_info('output', TensorProto.INT64, [2, None])

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
    output_tensor = helper.make_tensor_value_info('output', TensorProto.INT64, [2, None])

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
    output_tensor = helper.make_tensor_value_info('output', TensorProto.INT64, [2, None])

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
    output_tensor = helper.make_tensor_value_info('output', TensorProto.INT64, [1, None])

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
    output_tensor = helper.make_tensor_value_info('output', TensorProto.INT64, [3, None])

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
    output_tensor = helper.make_tensor_value_info('output', TensorProto.INT64, [2, None])

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

def validate_with_reference_evaluator():
    """Validate NonZero operation using ONNX ReferenceEvaluator"""
    print("\n=== ONNX ReferenceEvaluator Validation ===")

    # Test data that matches our Rust tests
    test_cases = [
        {
            'name': 'Float32 2D',
            'model_func': build_nonzero_float32_model,
            'input': np.array([
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 2.5, 0.0],
                [3.1, 0.0, 0.0, -1.2]
            ], dtype=np.float32)
        },
        {
            'name': 'Int64 2D',
            'model_func': build_nonzero_int64_model,
            'input': np.array([[5, 0, 0], [0, 0, -3]], dtype=np.int64)
        },
        {
            'name': 'Bool 2D',
            'model_func': build_nonzero_bool_model,
            'input': np.array([[False, True], [True, False]], dtype=bool)
        },
        {
            'name': '1D',
            'model_func': build_nonzero_1d_model,
            'input': np.array([0.0, 2.0, 0.0, -1.0, 0.0, 3.5], dtype=np.float32)
        },
        {
            'name': '3D',
            'model_func': build_nonzero_3d_model,
            'input': np.array([
                [[0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]]
            ], dtype=np.float32)
        },
        {
            'name': 'Empty (all zeros)',
            'model_func': build_nonzero_empty_model,
            'input': np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32)
        }
    ]

    for test_case in test_cases:
        print(f"\n{test_case['name']} Test:")
        print(f"Input shape: {test_case['input'].shape}")
        print(f"Input data:\n{test_case['input']}")

        # Build model and evaluate
        model = test_case['model_func']()
        evaluator = ReferenceEvaluator(model)
        result = evaluator.run(None, {'input': test_case['input']})

        output = result[0]
        print(f"Output shape: {output.shape}")
        print(f"Output data:\n{output}")
        print(f"Output format: [rank={output.shape[0]}, num_nonzero={output.shape[1]}]")

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
    print("- Output is always Int64 tensor with shape [input_rank, num_nonzero_elements]")
    print("- Each row contains all indices for one dimension")
    print("- Order follows C-style (row-major) traversal")

    # Validate with reference evaluator
    validate_with_reference_evaluator()

if __name__ == '__main__':
    main()