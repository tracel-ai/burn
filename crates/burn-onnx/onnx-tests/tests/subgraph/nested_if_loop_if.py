#!/usr/bin/env python3
"""
Generate ONNX model with 3 levels of nested subgraphs: If -> Loop -> If.
This tests recursive code generation and module hierarchy.

Structure:
- Outer If: condition1 determines which branch
  - Then branch: Contains a Loop
    - Loop body: Contains an inner If (condition2)
      - Inner If then: x + 1
      - Inner If else: x - 1
  - Else branch: Simple multiply by 2
"""

import onnx
from onnx import helper, TensorProto, numpy_helper
from onnx.reference import ReferenceEvaluator
import numpy as np


def build_model():
    """Build 3-level nested subgraph model."""

    # ========================================================================
    # Level 3 (innermost): If node inside Loop body
    # ========================================================================

    # Inner If - then branch: x + 1
    inner_then_add = helper.make_node(
        'Add',
        inputs=['x_inner', 'one_const'],
        outputs=['then_out']
    )
    inner_then_graph = helper.make_graph(
        nodes=[inner_then_add],
        name='inner_then',
        inputs=[helper.make_tensor_value_info('x_inner', TensorProto.FLOAT, [2, 3])],
        outputs=[helper.make_tensor_value_info('then_out', TensorProto.FLOAT, [2, 3])],
        initializer=[numpy_helper.from_array(np.array([1.0], dtype=np.float32), name='one_const')]
    )

    # Inner If - else branch: x - 1
    inner_else_sub = helper.make_node(
        'Sub',
        inputs=['x_inner', 'one_const'],
        outputs=['else_out']
    )
    inner_else_graph = helper.make_graph(
        nodes=[inner_else_sub],
        name='inner_else',
        inputs=[helper.make_tensor_value_info('x_inner', TensorProto.FLOAT, [2, 3])],
        outputs=[helper.make_tensor_value_info('else_out', TensorProto.FLOAT, [2, 3])],
        initializer=[numpy_helper.from_array(np.array([1.0], dtype=np.float32), name='one_const')]
    )

    # ========================================================================
    # Level 2: Loop body containing the inner If
    # ========================================================================

    # Loop body: contains If(condition2, then: x+1, else: x-1)
    # Inputs: [iter, cond, accum]
    # Outputs: [cond_out, accum_out]

    loop_if = helper.make_node(
        'If',
        inputs=['cond2'],
        outputs=['if_result'],
        then_branch=inner_then_graph,
        else_branch=inner_else_graph
    )

    # Condition is always true for simplicity
    loop_identity_cond = helper.make_node(
        'Identity',
        inputs=['cond_in'],
        outputs=['cond_out']
    )

    # Pass x to the inner if
    loop_identity_x = helper.make_node(
        'Identity',
        inputs=['accum'],
        outputs=['x_inner']
    )

    # Use if result as output
    loop_identity_out = helper.make_node(
        'Identity',
        inputs=['if_result'],
        outputs=['accum_out']
    )

    # Pass cond2 through unchanged (make it a loop-carried dependency)
    loop_identity_cond2 = helper.make_node(
        'Identity',
        inputs=['cond2'],
        outputs=['cond2_out']
    )

    loop_body_graph = helper.make_graph(
        nodes=[loop_identity_x, loop_if, loop_identity_cond, loop_identity_out, loop_identity_cond2],
        name='loop_body',
        inputs=[
            helper.make_tensor_value_info('iter', TensorProto.INT64, []),
            helper.make_tensor_value_info('cond_in', TensorProto.BOOL, []),  # Rank-0 (scalar)
            helper.make_tensor_value_info('accum', TensorProto.FLOAT, [2, 3]),
            helper.make_tensor_value_info('cond2', TensorProto.BOOL, []),  # Rank-0 (scalar)
        ],
        outputs=[
            helper.make_tensor_value_info('cond_out', TensorProto.BOOL, []),  # Rank-0 (scalar)
            helper.make_tensor_value_info('accum_out', TensorProto.FLOAT, [2, 3]),
            helper.make_tensor_value_info('cond2_out', TensorProto.BOOL, []),  # Rank-0 (scalar)
        ],
    )

    # ========================================================================
    # Level 1: Outer If - then branch contains Loop
    # ========================================================================

    # Loop takes cond2 as a loop-carried dependency (passed through unchanged)
    # ONNX spec requires: Loop must output all loop-carried dependencies
    outer_then_loop = helper.make_node(
        'Loop',
        inputs=['iterations', 'cond_init', 'x_init', 'condition2'],
        outputs=['loop_result', 'cond2_final'],  # Must output both loop-carried dependencies
        body=loop_body_graph
    )

    outer_then_graph = helper.make_graph(
        nodes=[outer_then_loop],
        name='outer_then',
        inputs=[
            helper.make_tensor_value_info('x_init', TensorProto.FLOAT, [2, 3]),
            helper.make_tensor_value_info('iterations', TensorProto.INT64, []),  # Rank-0 (scalar)
            helper.make_tensor_value_info('cond_init', TensorProto.BOOL, []),  # Rank-0 (scalar)
            # condition2 comes from outer scope (main graph)
        ],
        outputs=[
            helper.make_tensor_value_info('loop_result', TensorProto.FLOAT, [2, 3])
        ],
    )

    # ========================================================================
    # Level 1: Outer If - else branch (simple operation)
    # ========================================================================

    outer_else_mul = helper.make_node(
        'Mul',
        inputs=['x_init', 'two_const'],
        outputs=['else_result']
    )

    outer_else_graph = helper.make_graph(
        nodes=[outer_else_mul],
        name='outer_else',
        inputs=[helper.make_tensor_value_info('x_init', TensorProto.FLOAT, [2, 3])],
        outputs=[helper.make_tensor_value_info('else_result', TensorProto.FLOAT, [2, 3])],
        initializer=[numpy_helper.from_array(np.array([2.0], dtype=np.float32), name='two_const')]
    )

    # ========================================================================
    # Main graph: Outer If
    # ========================================================================

    # Identity nodes to make variables available to subgraphs
    identity_x = helper.make_node('Identity', inputs=['x'], outputs=['x_init'])
    identity_iter = helper.make_node('Identity', inputs=['M'], outputs=['iterations'])
    identity_cond = helper.make_node('Identity', inputs=['cond'], outputs=['cond_init'])

    # If only takes condition as input, other vars are from outer scope
    outer_if = helper.make_node(
        'If',
        inputs=['condition1'],
        outputs=['output'],
        then_branch=outer_then_graph,
        else_branch=outer_else_graph
    )

    main_graph = helper.make_graph(
        nodes=[identity_x, identity_iter, identity_cond, outer_if],
        name='nested_if_loop_if',
        inputs=[
            helper.make_tensor_value_info('x', TensorProto.FLOAT, [2, 3]),
            helper.make_tensor_value_info('M', TensorProto.INT64, []),  # Rank-0 (scalar)
            helper.make_tensor_value_info('cond', TensorProto.BOOL, []),  # Rank-0 (scalar)
            helper.make_tensor_value_info('condition1', TensorProto.BOOL, []),  # Rank-0 (scalar)
            helper.make_tensor_value_info('condition2', TensorProto.BOOL, []),  # Rank-0 (scalar)
        ],
        outputs=[
            helper.make_tensor_value_info('output', TensorProto.FLOAT, [2, 3])
        ],
    )

    # Create model
    model = helper.make_model(
        main_graph,
        producer_name='burn-onnx-test',
        opset_imports=[helper.make_opsetid("", 16)]
    )

    # Check model
    onnx.checker.check_model(model)
    return model


def generate_test_data(model):
    """Generate test data using ONNX reference evaluator."""

    x = np.random.randn(2, 3).astype(np.float32)
    M = np.array(3, dtype=np.int64)
    cond = np.array(True, dtype=bool)

    sess = ReferenceEvaluator(model)

    # Test case 1: condition1=True, condition2=True (then->loop->then: x+1, 3 times)
    out1 = sess.run(None, {
        'x': x,
        'M': M,
        'cond': cond,
        'condition1': np.array(True, dtype=bool),
        'condition2': np.array(True, dtype=bool),
    })[0]

    # Test case 2: condition1=True, condition2=False (then->loop->else: x-1, 3 times)
    out2 = sess.run(None, {
        'x': x,
        'M': M,
        'cond': cond,
        'condition1': np.array(True, dtype=bool),
        'condition2': np.array(False, dtype=bool),
    })[0]

    # Test case 3: condition1=False (else: x*2)
    out3 = sess.run(None, {
        'x': x,
        'M': M,
        'cond': cond,
        'condition1': np.array(False, dtype=bool),
        'condition2': np.array(True, dtype=bool),  # Doesn't matter
    })[0]

    return {
        'x': x,
        'M': M,
        'cond': cond,
        'test1_c1_true_c2_true': out1,
        'test2_c1_true_c2_false': out2,
        'test3_c1_false': out3,
    }


def main():
    """Generate model and test data."""

    model = build_model()

    # Save model
    onnx.save(model, 'nested_if_loop_if.onnx')
    print("✓ Saved nested_if_loop_if.onnx")

    # Generate test data
    test_data = generate_test_data(model)

    print("\n" + "="*80)
    print("Test data for nested_if_loop_if (3 levels of nesting):")
    print("="*80)

    print(f"\nInput x shape: {test_data['x'].shape}")
    print(f"Input x data: {test_data['x'].flatten().tolist()}")
    print(f"M (iterations): {test_data['M']}")
    print(f"cond: {test_data['cond']}")

    print(f"\n--- Test 1: condition1=True, condition2=True ---")
    print(f"Path: Outer Then -> Loop(3x) -> Inner Then (x+1 each iteration)")
    print(f"Expected: x + 3")
    print(f"Output: {test_data['test1_c1_true_c2_true'].flatten().tolist()}")
    expected1 = test_data['x'] + 3
    print(f"Manual calc: {expected1.flatten().tolist()}")
    print(f"Matches: {np.allclose(test_data['test1_c1_true_c2_true'], expected1)}")

    print(f"\n--- Test 2: condition1=True, condition2=False ---")
    print(f"Path: Outer Then -> Loop(3x) -> Inner Else (x-1 each iteration)")
    print(f"Expected: x - 3")
    print(f"Output: {test_data['test2_c1_true_c2_false'].flatten().tolist()}")
    expected2 = test_data['x'] - 3
    print(f"Manual calc: {expected2.flatten().tolist()}")
    print(f"Matches: {np.allclose(test_data['test2_c1_true_c2_false'], expected2)}")

    print(f"\n--- Test 3: condition1=False ---")
    print(f"Path: Outer Else (x*2)")
    print(f"Expected: x * 2")
    print(f"Output: {test_data['test3_c1_false'].flatten().tolist()}")
    expected3 = test_data['x'] * 2
    print(f"Manual calc: {expected3.flatten().tolist()}")
    print(f"Matches: {np.allclose(test_data['test3_c1_false'], expected3)}")

    print("\n" + "="*80)
    print("\nNesting structure:")
    print("Level 1: If(condition1)")
    print("  Then: Loop(M iterations)")
    print("    Level 2: Loop body")
    print("      Level 3: If(condition2)")
    print("        Then: x + 1")
    print("        Else: x - 1")
    print("  Else: x * 2")


if __name__ == '__main__':
    main()
