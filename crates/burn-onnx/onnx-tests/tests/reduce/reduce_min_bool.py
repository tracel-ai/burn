#!/usr/bin/env python3

"""
Test for ReduceMin with boolean tensors.

For boolean tensors, ReduceMin is equivalent to logical AND.
"""

import numpy as np
import onnx
from onnx import helper, TensorProto


def create_model():
    """Create ONNX model with ReduceMin operations on boolean tensors."""

    # Create input tensor (bool, shape [2, 3, 4])
    input_tensor = helper.make_tensor_value_info('input', TensorProto.BOOL, [2, 3, 4])

    # Output 1: Reduce all dimensions, no keepdims -> scalar bool
    reduce1 = helper.make_node(
        'ReduceMin',
        inputs=['input'],
        outputs=['output1'],
        keepdims=0,
    )

    # Output 2: Reduce all dimensions, keepdims=1 -> shape [1, 1, 1]
    reduce2 = helper.make_node(
        'ReduceMin',
        inputs=['input'],
        outputs=['output2'],
        keepdims=1,
    )

    # Output 3: Reduce along axis 2, no keepdims -> shape [2, 3]
    reduce3 = helper.make_node(
        'ReduceMin',
        inputs=['input'],
        outputs=['output3'],
        axes=[2],
        keepdims=0,
    )

    # Output 4: Reduce along axes [0, 2], keepdims=1 -> shape [1, 3, 1]
    reduce4 = helper.make_node(
        'ReduceMin',
        inputs=['input'],
        outputs=['output4'],
        axes=[0, 2],
        keepdims=1,
    )

    # Create output tensors
    output1_tensor = helper.make_tensor_value_info('output1', TensorProto.BOOL, [])
    output2_tensor = helper.make_tensor_value_info('output2', TensorProto.BOOL, [1, 1, 1])
    output3_tensor = helper.make_tensor_value_info('output3', TensorProto.BOOL, [2, 3])
    output4_tensor = helper.make_tensor_value_info('output4', TensorProto.BOOL, [1, 3, 1])

    # Create graph
    graph = helper.make_graph(
        nodes=[reduce1, reduce2, reduce3, reduce4],
        name='reduce_min_bool_model',
        inputs=[input_tensor],
        outputs=[output1_tensor, output2_tensor, output3_tensor, output4_tensor],
    )

    # Create model
    model = helper.make_model(
        graph,
        producer_name='burn-onnx-test',
        opset_imports=[helper.make_opsetid("", 16)]
    )

    # Check model
    onnx.checker.check_model(model)

    return model


def generate_test_data(model):
    """Generate test data and verify outputs using ReferenceEvaluator."""

    # Input data: [2, 3, 4] boolean tensor
    # Using a mix of True and False values
    input_data = np.array([
        [
            [True, True, False, True],   # All True except one
            [True, True, True, True],     # All True
            [False, False, False, False], # All False
        ],
        [
            [True, False, True, False],  # Mixed
            [True, True, True, False],   # Mostly True
            [False, True, False, True],  # Mixed
        ]
    ], dtype=bool)

    print("=" * 80)
    print("Test data for reduce_min_bool (ReduceMin on boolean = logical AND):")
    print("=" * 80)
    print()

    print("Input shape:", input_data.shape)
    print("Input data:")
    print(input_data)
    print()

    # Verify with ONNX ReferenceEvaluator
    try:
        from onnx.reference import ReferenceEvaluator

        sess = ReferenceEvaluator(model)
        outputs = sess.run(None, {"input": input_data})

        print("ONNX Model Outputs (using ReferenceEvaluator):")
        print()

        # Output 1: Reduce all -> scalar
        print("Output 1 (reduce all, no keepdims):")
        print(f"  Shape: {outputs[0].shape if hasattr(outputs[0], 'shape') else 'scalar'}")
        print(f"  Value: {outputs[0]}")
        print()

        # Output 2: Reduce all with keepdims
        print("Output 2 (reduce all, keepdims=1):")
        print(f"  Shape: {outputs[1].shape}")
        print(f"  Value: {outputs[1]}")
        print()

        # Output 3: Reduce along axis 2
        print("Output 3 (reduce axis 2, no keepdims):")
        print(f"  Shape: {outputs[2].shape}")
        print(f"  Value:")
        print(outputs[2])
        print()

        # Output 4: Reduce along axes [0, 2] with keepdims
        print("Output 4 (reduce axes [0, 2], keepdims=1):")
        print(f"  Shape: {outputs[3].shape}")
        print(f"  Value:")
        print(outputs[3])
        print()

    except ImportError:
        print("onnx.reference not available, falling back to NumPy verification")
        print()

        # Fallback to NumPy
        output1 = np.min(input_data)
        output2 = np.min(input_data, keepdims=True)
        output3 = np.min(input_data, axis=2, keepdims=False)
        output4 = np.min(input_data, axis=(0, 2), keepdims=True)

        print("NumPy Outputs:")
        print(f"Output 1: {output1}")
        print(f"Output 2: {output2}")
        print(f"Output 3: {output3}")
        print(f"Output 4: {output4}")
        print()

    print("=" * 80)
    print("Explanation:")
    print("- For booleans, ReduceMin is equivalent to logical AND")
    print("- Output 1: AND of all 24 elements = False (there are False values)")
    print("- Output 3: AND along last axis for each [2, 3] position")
    print("  - [0,0,:] = True AND True AND False AND True = False")
    print("  - [0,1,:] = All True = True")
    print("  - [0,2,:] = All False = False")
    print("  - etc.")
    print("=" * 80)


if __name__ == '__main__':
    model = create_model()

    # Save model
    onnx.save(model, 'reduce_min_bool.onnx')
    print("✓ Saved reduce_min_bool.onnx")
    print()

    # Generate test data
    generate_test_data(model)
