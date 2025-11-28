#!/usr/bin/env python3
"""
Generate ONNX model with 4 levels of nested subgraphs: If -> Loop -> If -> Scan.
This tests deep recursive code generation and module hierarchy.

Structure:
- Outer If: condition1 determines which branch
  - Then branch: Contains a Loop
    - Loop body: Contains an inner If (condition2)
      - Inner If then: Contains a Scan (cumulative sum)
      - Inner If else: x - 1
  - Else branch: Simple multiply by 2
"""

import onnx
from onnx import helper, TensorProto, numpy_helper
from onnx.reference import ReferenceEvaluator
import numpy as np


def build_model():
    """Build 4-level nested subgraph model."""

    # ========================================================================
    # Level 4 (innermost): Scan node inside inner If's then branch
    # ========================================================================

    # Scan body: cumulative sum along sequence
    # Inputs: [sum_state, elem]
    # Outputs: [sum_out]
    scan_add = helper.make_node(
        'Add',
        inputs=['sum_in', 'elem'],
        outputs=['sum_out']
    )

    scan_body_graph = helper.make_graph(
        nodes=[scan_add],
        name='scan_body',
        inputs=[
            helper.make_tensor_value_info('sum_in', TensorProto.FLOAT, [3]),
            helper.make_tensor_value_info('elem', TensorProto.FLOAT, [3]),
        ],
        outputs=[
            helper.make_tensor_value_info('sum_out', TensorProto.FLOAT, [3]),
        ],
    )

    # ========================================================================
    # Level 3: Inner If - then branch contains Scan
    # ========================================================================

    # Prepare sequence for Scan: split x_inner into 2 slices along axis 0
    # x_inner is [2, 3], we need to make it [2, 3] -> treat as 2 timesteps of shape [3]

    # Identity to pass x_inner to Scan as sequence
    inner_then_identity = helper.make_node(
        'Identity',
        inputs=['x_inner'],
        outputs=['scan_sequence']
    )

    # Initial sum state (zeros)
    # Scan node
    inner_then_scan = helper.make_node(
        'Scan',
        inputs=['sum_init', 'scan_sequence'],
        outputs=['final_sum'],
        num_scan_inputs=1,
        body=scan_body_graph
    )

    # Use zeros as initial sum state (hardcoded in the graph)
    sum_init_const = np.zeros(3, dtype=np.float32)

    inner_then_graph = helper.make_graph(
        nodes=[inner_then_identity, inner_then_scan],
        name='inner_then',
        inputs=[
            helper.make_tensor_value_info('x_inner', TensorProto.FLOAT, [2, 3]),
        ],
        outputs=[helper.make_tensor_value_info('final_sum', TensorProto.FLOAT, [3])],
        initializer=[
            numpy_helper.from_array(sum_init_const, name='sum_init'),
        ]
    )

    # ========================================================================
    # Level 3: Inner If - else branch (simple operation)
    # ========================================================================

    inner_else_sub = helper.make_node(
        'Sub',
        inputs=['x_inner', 'one_const'],
        outputs=['else_out']
    )

    # Reduce to match output shape [3] instead of [2, 3]
    # In opset 16, axes is an input, not an attribute
    axes_const = np.array([0], dtype=np.int64)
    inner_else_reduce = helper.make_node(
        'ReduceSum',
        inputs=['else_out', 'axes_const'],
        outputs=['else_reduced'],
        keepdims=0
    )

    inner_else_graph = helper.make_graph(
        nodes=[inner_else_sub, inner_else_reduce],
        name='inner_else',
        inputs=[helper.make_tensor_value_info('x_inner', TensorProto.FLOAT, [2, 3])],
        outputs=[helper.make_tensor_value_info('else_reduced', TensorProto.FLOAT, [3])],
        initializer=[
            numpy_helper.from_array(np.array([1.0], dtype=np.float32), name='one_const'),
            numpy_helper.from_array(axes_const, name='axes_const'),
        ]
    )

    # ========================================================================
    # Level 2: Loop body containing the inner If
    # ========================================================================

    # Loop body: contains If(condition2, then: Scan(x), else: sum(x-1))
    # Inputs: [iter, cond, accum, cond2]
    # Outputs: [cond_out, accum_out, cond2_out]
    # Note: sum_init is accessed from outer scope (not loop-carried since it's never modified)

    loop_if = helper.make_node(
        'If',
        inputs=['cond2'],
        outputs=['if_result'],
        name='inner_if',  # Unique name to avoid collision with outer If
        then_branch=inner_then_graph,
        else_branch=inner_else_graph
    )

    # Condition is always true for simplicity
    loop_identity_cond = helper.make_node(
        'Identity',
        inputs=['cond_in'],
        outputs=['cond_out']
    )

    # Pass accum to the inner if as x_inner
    loop_identity_x = helper.make_node(
        'Identity',
        inputs=['accum'],
        outputs=['x_inner']
    )

    # Pass cond2 through unchanged
    loop_identity_cond2 = helper.make_node(
        'Identity',
        inputs=['cond2'],
        outputs=['cond2_out']
    )

    # Expand if_result back to [2, 3] by adding zeros
    # if_result is [3], we need [2, 3]
    # Use Unsqueeze + Concat
    unsqueeze_axes = np.array([0], dtype=np.int64)
    loop_unsqueeze = helper.make_node(
        'Unsqueeze',
        inputs=['if_result', 'unsqueeze_axes'],
        outputs=['if_result_unsqueezed']
    )

    loop_concat = helper.make_node(
        'Concat',
        inputs=['if_result_unsqueezed', 'if_result_unsqueezed'],
        outputs=['accum_out'],
        axis=0
    )

    loop_body_graph = helper.make_graph(
        nodes=[loop_identity_x, loop_if, loop_identity_cond, loop_unsqueeze, loop_concat, loop_identity_cond2],
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
        initializer=[
            numpy_helper.from_array(unsqueeze_axes, name='unsqueeze_axes'),
        ]
    )

    # ========================================================================
    # Level 1: Outer If - then branch contains Loop
    # ========================================================================

    # ONNX spec requires: Loop must output all loop-carried dependencies
    outer_then_loop = helper.make_node(
        'Loop',
        inputs=['iterations', 'cond_init', 'x_init', 'condition2'],
        outputs=['loop_result', 'cond2_final'],  # Must output all 2 loop-carried dependencies
        body=loop_body_graph
    )

    outer_then_graph = helper.make_graph(
        nodes=[outer_then_loop],
        name='outer_then',
        inputs=[
            helper.make_tensor_value_info('x_init', TensorProto.FLOAT, [2, 3]),
            helper.make_tensor_value_info('iterations', TensorProto.INT64, []),  # Rank-0 (scalar)
            helper.make_tensor_value_info('cond_init', TensorProto.BOOL, []),  # Rank-0 (scalar)
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
        name='outer_if',  # Unique name to distinguish from inner If
        then_branch=outer_then_graph,
        else_branch=outer_else_graph
    )

    main_graph = helper.make_graph(
        nodes=[identity_x, identity_iter, identity_cond, outer_if],
        name='nested_if_loop_if_scan',
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
        producer_name='burn-import-test',
        opset_imports=[helper.make_opsetid("", 16)]
    )

    # Check model
    onnx.checker.check_model(model)
    return model


def generate_test_data(model):
    """Generate test data using ONNX reference evaluator."""

    x = np.random.randn(2, 3).astype(np.float32)
    M = np.array(2, dtype=np.int64)
    cond = np.array(True, dtype=bool)

    sess = ReferenceEvaluator(model)

    # Test case 1: condition1=True, condition2=True (then->loop->then->scan)
    out1 = sess.run(None, {
        'x': x,
        'M': M,
        'cond': cond,
        'condition1': np.array(True, dtype=bool),
        'condition2': np.array(True, dtype=bool),
    })[0]

    # Test case 2: condition1=True, condition2=False (then->loop->else)
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
        'condition2': np.array(True, dtype=bool),
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
    onnx.save(model, 'nested_if_loop_if_scan.onnx')
    print("✓ Saved nested_if_loop_if_scan.onnx")

    # Generate test data
    test_data = generate_test_data(model)

    print("\n" + "="*80)
    print("Test data for nested_if_loop_if_scan (4 levels of nesting):")
    print("="*80)

    print(f"\nInput x shape: {test_data['x'].shape}")
    print(f"Input x data: {test_data['x'].flatten().tolist()}")
    print(f"M (iterations): {test_data['M']}")
    print(f"cond: {test_data['cond']}")

    print(f"\n--- Test 1: condition1=True, condition2=True ---")
    print(f"Path: Outer Then -> Loop(2x) -> Inner Then -> Scan (cumsum)")
    print(f"Output: {test_data['test1_c1_true_c2_true'].flatten().tolist()}")

    print(f"\n--- Test 2: condition1=True, condition2=False ---")
    print(f"Path: Outer Then -> Loop(2x) -> Inner Else (sum(x-1))")
    print(f"Output: {test_data['test2_c1_true_c2_false'].flatten().tolist()}")

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
    print("        Then: Scan (cumulative sum)")
    print("          Level 4: Scan body (add)")
    print("        Else: sum(x - 1)")
    print("  Else: x * 2")


if __name__ == '__main__':
    main()
