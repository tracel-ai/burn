#!/usr/bin/env python3
"""
Generate ONNX model with deeply nested If operators (3-4 levels).
Tests that unique naming works across multiple levels of nested subgraphs.
"""

import onnx
from onnx import helper, TensorProto, numpy_helper
from onnx.reference import ReferenceEvaluator
import numpy as np


def build_model():
    """Build ONNX model with 4 levels of nested If operators."""

    batch_size = 2
    input_size = 3

    # Input
    x = np.random.randn(batch_size, input_size).astype(np.float32)

    # ============================================================================
    # LEVEL 4: Innermost If nodes (leaf operations)
    # ============================================================================

    # Level 4a: Then-Then-Then branch (add constant)
    const_4a = np.array([1.0], dtype=np.float32)
    level4a_add = helper.make_node('Add', inputs=['x_input', 'const_4a'], outputs=['level4a_out'])
    level4a_graph = helper.make_graph(
        nodes=[level4a_add],
        name='level4a_then_then_then',
        inputs=[helper.make_tensor_value_info('x_input', TensorProto.FLOAT, [batch_size, input_size])],
        outputs=[helper.make_tensor_value_info('level4a_out', TensorProto.FLOAT, [batch_size, input_size])],
        initializer=[numpy_helper.from_array(const_4a, name='const_4a')]
    )

    # Level 4b: Then-Then-Else branch (multiply constant)
    const_4b = np.array([2.0], dtype=np.float32)
    level4b_mul = helper.make_node('Mul', inputs=['x_input', 'const_4b'], outputs=['level4b_out'])
    level4b_graph = helper.make_graph(
        nodes=[level4b_mul],
        name='level4b_then_then_else',
        inputs=[helper.make_tensor_value_info('x_input', TensorProto.FLOAT, [batch_size, input_size])],
        outputs=[helper.make_tensor_value_info('level4b_out', TensorProto.FLOAT, [batch_size, input_size])],
        initializer=[numpy_helper.from_array(const_4b, name='const_4b')]
    )

    # ============================================================================
    # LEVEL 3: Second level If nodes
    # ============================================================================

    # Level 3a: Then-Then branch (contains Level 4a/4b)
    level3a_if = helper.make_node(
        'If',
        inputs=['cond3'],
        outputs=['level3a_out'],
        then_branch=level4a_graph,
        else_branch=level4b_graph,
    )
    level3a_graph = helper.make_graph(
        nodes=[level3a_if],
        name='level3a_then_then',
        inputs=[
            helper.make_tensor_value_info('x_input', TensorProto.FLOAT, [batch_size, input_size]),
            helper.make_tensor_value_info('cond3', TensorProto.BOOL, []),
        ],
        outputs=[helper.make_tensor_value_info('level3a_out', TensorProto.FLOAT, [batch_size, input_size])],
    )

    # Level 3b: Then-Else branch (subtract constant)
    const_3b = np.array([0.5], dtype=np.float32)
    level3b_sub = helper.make_node('Sub', inputs=['x_input', 'const_3b'], outputs=['level3b_out'])
    level3b_graph = helper.make_graph(
        nodes=[level3b_sub],
        name='level3b_then_else',
        inputs=[helper.make_tensor_value_info('x_input', TensorProto.FLOAT, [batch_size, input_size])],
        outputs=[helper.make_tensor_value_info('level3b_out', TensorProto.FLOAT, [batch_size, input_size])],
        initializer=[numpy_helper.from_array(const_3b, name='const_3b')]
    )

    # Level 3c: Else-Then branch (divide constant)
    const_3c = np.array([3.0], dtype=np.float32)
    level3c_div = helper.make_node('Div', inputs=['x_input', 'const_3c'], outputs=['level3c_out'])
    level3c_graph = helper.make_graph(
        nodes=[level3c_div],
        name='level3c_else_then',
        inputs=[helper.make_tensor_value_info('x_input', TensorProto.FLOAT, [batch_size, input_size])],
        outputs=[helper.make_tensor_value_info('level3c_out', TensorProto.FLOAT, [batch_size, input_size])],
        initializer=[numpy_helper.from_array(const_3c, name='const_3c')]
    )

    # Level 3d: Else-Else branch (negate)
    level3d_neg = helper.make_node('Neg', inputs=['x_input'], outputs=['level3d_out'])
    level3d_graph = helper.make_graph(
        nodes=[level3d_neg],
        name='level3d_else_else',
        inputs=[helper.make_tensor_value_info('x_input', TensorProto.FLOAT, [batch_size, input_size])],
        outputs=[helper.make_tensor_value_info('level3d_out', TensorProto.FLOAT, [batch_size, input_size])],
    )

    # ============================================================================
    # LEVEL 2: First level If nodes
    # ============================================================================

    # Level 2a: Then branch (contains Level 3a/3b)
    level2a_if = helper.make_node(
        'If',
        inputs=['cond2'],
        outputs=['level2a_out'],
        then_branch=level3a_graph,
        else_branch=level3b_graph,
    )
    level2a_graph = helper.make_graph(
        nodes=[level2a_if],
        name='level2a_then',
        inputs=[
            helper.make_tensor_value_info('x_input', TensorProto.FLOAT, [batch_size, input_size]),
            helper.make_tensor_value_info('cond2', TensorProto.BOOL, []),
            helper.make_tensor_value_info('cond3', TensorProto.BOOL, []),
        ],
        outputs=[helper.make_tensor_value_info('level2a_out', TensorProto.FLOAT, [batch_size, input_size])],
    )

    # Level 2b: Else branch (contains Level 3c/3d)
    level2b_if = helper.make_node(
        'If',
        inputs=['cond2'],
        outputs=['level2b_out'],
        then_branch=level3c_graph,
        else_branch=level3d_graph,
    )
    level2b_graph = helper.make_graph(
        nodes=[level2b_if],
        name='level2b_else',
        inputs=[
            helper.make_tensor_value_info('x_input', TensorProto.FLOAT, [batch_size, input_size]),
            helper.make_tensor_value_info('cond2', TensorProto.BOOL, []),
        ],
        outputs=[helper.make_tensor_value_info('level2b_out', TensorProto.FLOAT, [batch_size, input_size])],
    )

    # ============================================================================
    # LEVEL 1: Main graph
    # ============================================================================

    # Main If node
    level1_if = helper.make_node(
        'If',
        inputs=['cond1'],
        outputs=['output'],
        then_branch=level2a_graph,
        else_branch=level2b_graph,
    )

    # Use Identity node to make 'x' available to subgraphs as 'x_input'
    identity = helper.make_node('Identity', inputs=['x'], outputs=['x_input'])

    # Main graph
    graph = helper.make_graph(
        nodes=[identity, level1_if],
        name='nested_if_model',
        inputs=[
            helper.make_tensor_value_info('x', TensorProto.FLOAT, [batch_size, input_size]),
            helper.make_tensor_value_info('cond1', TensorProto.BOOL, []),
            helper.make_tensor_value_info('cond2', TensorProto.BOOL, []),
            helper.make_tensor_value_info('cond3', TensorProto.BOOL, []),
        ],
        outputs=[
            helper.make_tensor_value_info('output', TensorProto.FLOAT, [batch_size, input_size])
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
    """Generate test inputs and expected outputs using ONNX reference evaluator."""

    batch_size = 2
    input_size = 3

    # Input tensor
    x = np.random.randn(batch_size, input_size).astype(np.float32)

    # Create reference evaluator
    sess = ReferenceEvaluator(model)

    # Test different paths through the nested structure
    test_cases = [
        # (cond1, cond2, cond3, description)
        (True, True, True, "then_then_then"),      # Level 4a: x + 1.0
        (True, True, False, "then_then_else"),     # Level 4b: x * 2.0
        (True, False, None, "then_else"),          # Level 3b: x - 0.5
        (False, True, None, "else_then"),          # Level 3c: x / 3.0
        (False, False, None, "else_else"),         # Level 3d: -x
    ]

    results = {}
    for cond1, cond2, cond3, desc in test_cases:
        inputs = {
            'x': x,
            'cond1': np.array(cond1),
            'cond2': np.array(cond2 if cond2 is not None else False),
            'cond3': np.array(cond3 if cond3 is not None else False),
        }
        output = sess.run(None, inputs)[0]
        results[desc] = output

    return {'input': x, 'outputs': results}


def main():
    """Generate model and test data."""

    # Build model
    model = build_model()

    # Save model
    onnx.save(model, 'nested_if.onnx')
    print("✓ Saved nested_if.onnx")

    # Generate test data using ONNX reference implementation
    test_data = generate_test_data(model)

    # Print test data for copying into Rust tests
    print("\n" + "="*80)
    print("Test data for nested_if:")
    print("="*80)

    print("\nInput tensor:")
    print(f"Shape: {test_data['input'].shape}")
    print(f"Data: {test_data['input'].flatten().tolist()}")

    for path, output in test_data['outputs'].items():
        print(f"\nPath '{path}' output:")
        print(f"Shape: {output.shape}")
        print(f"Data: {output.flatten().tolist()}")

    print("\n" + "="*80)


if __name__ == '__main__':
    main()
