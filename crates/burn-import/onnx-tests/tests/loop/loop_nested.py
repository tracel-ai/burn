#!/usr/bin/env python3
"""
Generate ONNX model with nested Loop operators.
Tests outer loop containing an inner loop.

Outer loop: Iterates M_outer times, accumulating a sum
Inner loop: For each outer iteration, runs M_inner times, adding a constant
"""

import onnx
from onnx import helper, TensorProto, numpy_helper
from onnx.reference import ReferenceEvaluator
import numpy as np


def build_model():
    """Build ONNX model with nested Loop operators."""

    # Inner loop body: sum_inner = sum_inner + constant
    # Inner loop inputs: [iter_num, cond, sum_inner]
    # Inner loop outputs: [cond, sum_inner]

    inner_const = np.array([3.0], dtype=np.float32)

    # Inner loop body nodes
    inner_add = helper.make_node(
        'Add',
        inputs=['sum_inner', 'inner_const'],
        outputs=['sum_inner_out']
    )

    inner_identity_cond = helper.make_node(
        'Identity',
        inputs=['cond_inner'],
        outputs=['cond_inner_out']
    )

    # Create inner loop body graph
    inner_body_graph = helper.make_graph(
        nodes=[inner_add, inner_identity_cond],
        name='inner_loop_body',
        inputs=[
            helper.make_tensor_value_info('iter_inner', TensorProto.INT64, []),
            helper.make_tensor_value_info('cond_inner', TensorProto.BOOL, []),
            helper.make_tensor_value_info('sum_inner', TensorProto.FLOAT, [1]),
        ],
        outputs=[
            helper.make_tensor_value_info('cond_inner_out', TensorProto.BOOL, []),
            helper.make_tensor_value_info('sum_inner_out', TensorProto.FLOAT, [1]),
        ],
        initializer=[
            numpy_helper.from_array(inner_const, name='inner_const'),
        ]
    )

    # Outer loop body contains the inner loop
    # Outer loop inputs: [iter_num, cond, sum_outer, m_inner_val]
    # Outer loop outputs: [cond, sum_outer, m_inner_val]
    # m_inner_val is a loop-carried dependency that doesn't change

    outer_const = np.array([1.0], dtype=np.float32)

    # Inner loop node in outer body
    inner_loop = helper.make_node(
        'Loop',
        inputs=['m_inner_val', 'cond_outer', 'sum_outer'],  # max_iterations, initial_cond, loop_var
        outputs=['sum_after_inner'],
        body=inner_body_graph
    )

    # Add outer_const to result of inner loop
    outer_add = helper.make_node(
        'Add',
        inputs=['sum_after_inner', 'outer_const'],
        outputs=['sum_outer_out']
    )

    outer_identity_cond = helper.make_node(
        'Identity',
        inputs=['cond_outer'],
        outputs=['cond_outer_out']
    )

    # Pass m_inner_val through unchanged
    outer_identity_m_inner = helper.make_node(
        'Identity',
        inputs=['m_inner_val'],
        outputs=['m_inner_val_out']
    )

    # Create outer loop body graph
    outer_body_graph = helper.make_graph(
        nodes=[inner_loop, outer_add, outer_identity_cond, outer_identity_m_inner],
        name='outer_loop_body',
        inputs=[
            helper.make_tensor_value_info('iter_outer', TensorProto.INT64, []),
            helper.make_tensor_value_info('cond_outer', TensorProto.BOOL, []),
            helper.make_tensor_value_info('sum_outer', TensorProto.FLOAT, [1]),
            helper.make_tensor_value_info('m_inner_val', TensorProto.INT64, []),
        ],
        outputs=[
            helper.make_tensor_value_info('cond_outer_out', TensorProto.BOOL, []),
            helper.make_tensor_value_info('sum_outer_out', TensorProto.FLOAT, [1]),
            helper.make_tensor_value_info('m_inner_val_out', TensorProto.INT64, []),
        ],
        initializer=[
            numpy_helper.from_array(outer_const, name='outer_const'),
        ]
    )

    # Main graph with outer loop
    # M_inner is passed as a loop-carried dependency (doesn't change)
    outer_loop = helper.make_node(
        'Loop',
        inputs=['M_outer', 'cond_init', 'sum_init', 'M_inner'],
        outputs=['sum_final', 'm_inner_final']
    )

    # Set the body attribute
    outer_loop.attribute.append(
        helper.make_attribute('body', outer_body_graph)
    )

    # Create main graph
    graph = helper.make_graph(
        nodes=[outer_loop],
        name='nested_loop_model',
        inputs=[
            helper.make_tensor_value_info('M_outer', TensorProto.INT64, []),
            helper.make_tensor_value_info('M_inner', TensorProto.INT64, []),
            helper.make_tensor_value_info('cond_init', TensorProto.BOOL, []),
            helper.make_tensor_value_info('sum_init', TensorProto.FLOAT, [1]),
        ],
        outputs=[
            helper.make_tensor_value_info('sum_final', TensorProto.FLOAT, [1]),
            helper.make_tensor_value_info('m_inner_final', TensorProto.INT64, []),
        ],
    )

    # Create model
    model = helper.make_model(
        graph,
        producer_name='burn-import-test',
        opset_imports=[helper.make_opsetid("", 16)]
    )

    # Check model
    onnx.checker.check_model(model)

    return model


def generate_test_data(model):
    """Generate test data and verify outputs using ReferenceEvaluator."""

    print("=" * 80)
    print("Test data for nested loop:")
    print("=" * 80)
    print()
    print("Structure:")
    print("  Outer loop: Runs M_outer times")
    print("  Inner loop: Runs M_inner times (inside each outer iteration)")
    print("  Inner loop: sum = sum + 3.0 each iteration")
    print("  Outer loop: sum = (result from inner) + 1.0 each iteration")
    print()

    # Test case 1: Outer=2, Inner=3
    M_outer = np.array(2, dtype=np.int64)
    M_inner = np.array(3, dtype=np.int64)
    cond_init = np.array(True, dtype=bool)
    sum_init = np.array([0.0], dtype=np.float32)

    print(f"Test 1: M_outer={M_outer}, M_inner={M_inner}, sum_init={sum_init[0]}")
    print()

    # Manual calculation:
    # Initial: sum = 0
    # Outer iteration 1:
    #   Inner: 0 + 3 + 3 + 3 = 9
    #   Outer: 9 + 1 = 10
    # Outer iteration 2:
    #   Inner: 10 + 3 + 3 + 3 = 19
    #   Outer: 19 + 1 = 20
    # Final: 20

    try:
        from onnx.reference import ReferenceEvaluator

        sess = ReferenceEvaluator(model)
        outputs = sess.run(None, {
            "M_outer": M_outer,
            "M_inner": M_inner,
            "cond_init": cond_init,
            "sum_init": sum_init
        })

        print("ONNX Model Output (using ReferenceEvaluator):")
        print(f"  sum_final: {outputs[0]}")
        print(f"  Expected:  [20.0]")
        print()

        # Verify
        expected = np.array([20.0], dtype=np.float32)
        np.testing.assert_allclose(outputs[0], expected, rtol=1e-5)
        print("✓ Test 1 passed!")
        print()

    except ImportError:
        print("onnx.reference not available, skipping ONNX model verification")
        print("Expected output: [20.0]")
        print()

    # Test case 2: Outer=3, Inner=2
    M_outer = np.array(3, dtype=np.int64)
    M_inner = np.array(2, dtype=np.int64)
    sum_init = np.array([5.0], dtype=np.float32)

    print(f"Test 2: M_outer={M_outer}, M_inner={M_inner}, sum_init={sum_init[0]}")
    print()

    # Manual calculation:
    # Initial: sum = 5
    # Outer iteration 1:
    #   Inner: 5 + 3 + 3 = 11
    #   Outer: 11 + 1 = 12
    # Outer iteration 2:
    #   Inner: 12 + 3 + 3 = 18
    #   Outer: 18 + 1 = 19
    # Outer iteration 3:
    #   Inner: 19 + 3 + 3 = 25
    #   Outer: 25 + 1 = 26
    # Final: 26

    try:
        from onnx.reference import ReferenceEvaluator

        sess = ReferenceEvaluator(model)
        outputs = sess.run(None, {
            "M_outer": M_outer,
            "M_inner": M_inner,
            "cond_init": cond_init,
            "sum_init": sum_init
        })

        print("ONNX Model Output (using ReferenceEvaluator):")
        print(f"  sum_final: {outputs[0]}")
        print(f"  Expected:  [26.0]")
        print()

        # Verify
        expected = np.array([26.0], dtype=np.float32)
        np.testing.assert_allclose(outputs[0], expected, rtol=1e-5)
        print("✓ Test 2 passed!")
        print()

    except ImportError:
        print("Expected output: [26.0]")
        print()

    print("=" * 80)
    print("Explanation:")
    print("- Each outer iteration runs the inner loop M_inner times")
    print("- Inner loop adds 3.0 per iteration")
    print("- After inner loop completes, outer adds 1.0")
    print("- Total added per outer iteration: (M_inner * 3.0) + 1.0")
    print("=" * 80)


if __name__ == '__main__':
    model = build_model()

    # Save model
    onnx.save(model, 'loop_nested.onnx')
    print("✓ Saved loop_nested.onnx")
    print()

    # Generate test data
    generate_test_data(model)
