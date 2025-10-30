#!/usr/bin/env python3
"""
Generate ONNX model with deeply nested If operators testing variable scoping.
Tests that variables can be passed from outer scopes through nested subgraphs.

Pattern:
  Root: var1 (x)
    Level 1: uses var1, produces var2 (y)
      Level 2: uses var2, produces var3 (z)
        Level 3: uses var3, produces final output
"""

import onnx
from onnx import helper, TensorProto, numpy_helper
from onnx.reference import ReferenceEvaluator
import numpy as np


def build_model():
    """Build ONNX model with nested If operators testing variable scoping."""

    batch_size = 2
    input_size = 3

    # ============================================================================
    # LEVEL 3: Innermost If node (uses var3 from parent)
    # ============================================================================

    # Level 3 Then: var3 + 1.0
    const_3_then = np.array([1.0], dtype=np.float32)
    level3_then_add = helper.make_node('Add', inputs=['var3', 'const_3_then'], outputs=['level3_then_out'])
    level3_then_graph = helper.make_graph(
        nodes=[level3_then_add],
        name='level3_then',
        inputs=[helper.make_tensor_value_info('var3', TensorProto.FLOAT, [batch_size, input_size])],
        outputs=[helper.make_tensor_value_info('level3_then_out', TensorProto.FLOAT, [batch_size, input_size])],
        initializer=[numpy_helper.from_array(const_3_then, name='const_3_then')]
    )

    # Level 3 Else: var3 * 2.0
    const_3_else = np.array([2.0], dtype=np.float32)
    level3_else_mul = helper.make_node('Mul', inputs=['var3', 'const_3_else'], outputs=['level3_else_out'])
    level3_else_graph = helper.make_graph(
        nodes=[level3_else_mul],
        name='level3_else',
        inputs=[helper.make_tensor_value_info('var3', TensorProto.FLOAT, [batch_size, input_size])],
        outputs=[helper.make_tensor_value_info('level3_else_out', TensorProto.FLOAT, [batch_size, input_size])],
        initializer=[numpy_helper.from_array(const_3_else, name='const_3_else')]
    )

    # ============================================================================
    # LEVEL 2: Middle If node (uses var2, produces var3, contains level 3)
    # ============================================================================

    # Level 2 Then: produces var3 = var2 - 0.5, then passes to level 3
    const_2_then = np.array([0.5], dtype=np.float32)
    level2_then_sub = helper.make_node('Sub', inputs=['var2', 'const_2_then'], outputs=['var3'])
    level2_then_if = helper.make_node(
        'If',
        inputs=['cond3'],
        outputs=['level2_then_out'],
        then_branch=level3_then_graph,
        else_branch=level3_else_graph,
    )
    level2_then_graph = helper.make_graph(
        nodes=[level2_then_sub, level2_then_if],
        name='level2_then',
        inputs=[
            helper.make_tensor_value_info('var2', TensorProto.FLOAT, [batch_size, input_size]),
            helper.make_tensor_value_info('cond3', TensorProto.BOOL, []),
        ],
        outputs=[helper.make_tensor_value_info('level2_then_out', TensorProto.FLOAT, [batch_size, input_size])],
        initializer=[numpy_helper.from_array(const_2_then, name='const_2_then')]
    )

    # Level 2 Else: produces var3 = var2 / 3.0, then passes to level 3
    const_2_else = np.array([3.0], dtype=np.float32)
    level2_else_div = helper.make_node('Div', inputs=['var2', 'const_2_else'], outputs=['var3'])
    level2_else_if = helper.make_node(
        'If',
        inputs=['cond3'],
        outputs=['level2_else_out'],
        then_branch=level3_then_graph,
        else_branch=level3_else_graph,
    )
    level2_else_graph = helper.make_graph(
        nodes=[level2_else_div, level2_else_if],
        name='level2_else',
        inputs=[
            helper.make_tensor_value_info('var2', TensorProto.FLOAT, [batch_size, input_size]),
            helper.make_tensor_value_info('cond3', TensorProto.BOOL, []),
        ],
        outputs=[helper.make_tensor_value_info('level2_else_out', TensorProto.FLOAT, [batch_size, input_size])],
        initializer=[numpy_helper.from_array(const_2_else, name='const_2_else')]
    )

    # ============================================================================
    # LEVEL 1: Outer If node (uses var1, produces var2, contains level 2)
    # ============================================================================

    # Level 1 Then: produces var2 = var1 + 10.0, then passes to level 2
    const_1_then = np.array([10.0], dtype=np.float32)
    level1_then_add = helper.make_node('Add', inputs=['var1', 'const_1_then'], outputs=['var2'])
    level1_then_if = helper.make_node(
        'If',
        inputs=['cond2'],
        outputs=['level1_then_out'],
        then_branch=level2_then_graph,
        else_branch=level2_else_graph,
    )
    level1_then_graph = helper.make_graph(
        nodes=[level1_then_add, level1_then_if],
        name='level1_then',
        inputs=[
            helper.make_tensor_value_info('var1', TensorProto.FLOAT, [batch_size, input_size]),
            helper.make_tensor_value_info('cond2', TensorProto.BOOL, []),
            helper.make_tensor_value_info('cond3', TensorProto.BOOL, []),
        ],
        outputs=[helper.make_tensor_value_info('level1_then_out', TensorProto.FLOAT, [batch_size, input_size])],
        initializer=[numpy_helper.from_array(const_1_then, name='const_1_then')]
    )

    # Level 1 Else: produces var2 = -var1, then passes to level 2
    level1_else_neg = helper.make_node('Neg', inputs=['var1'], outputs=['var2'])
    level1_else_if = helper.make_node(
        'If',
        inputs=['cond2'],
        outputs=['level1_else_out'],
        then_branch=level2_then_graph,
        else_branch=level2_else_graph,
    )
    level1_else_graph = helper.make_graph(
        nodes=[level1_else_neg, level1_else_if],
        name='level1_else',
        inputs=[
            helper.make_tensor_value_info('var1', TensorProto.FLOAT, [batch_size, input_size]),
            helper.make_tensor_value_info('cond2', TensorProto.BOOL, []),
            helper.make_tensor_value_info('cond3', TensorProto.BOOL, []),
        ],
        outputs=[helper.make_tensor_value_info('level1_else_out', TensorProto.FLOAT, [batch_size, input_size])],
    )

    # ============================================================================
    # ROOT: Main graph
    # ============================================================================

    # Main If node
    main_if = helper.make_node(
        'If',
        inputs=['cond1'],
        outputs=['output'],
        then_branch=level1_then_graph,
        else_branch=level1_else_graph,
    )

    # Identity to create var1 from input x
    identity = helper.make_node('Identity', inputs=['x'], outputs=['var1'])

    # Main graph
    graph = helper.make_graph(
        nodes=[identity, main_if],
        name='nested_if_scoped_vars',
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
    x = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0]
    ], dtype=np.float32)

    # Create reference evaluator
    sess = ReferenceEvaluator(model)

    # Test different paths through the nested structure
    test_cases = [
        # (cond1, cond2, cond3, description, computation)
        (True, True, True, "then_then_then", "((x + 10) - 0.5) + 1.0"),
        (True, True, False, "then_then_else", "((x + 10) - 0.5) * 2.0"),
        (True, False, True, "then_else_then", "((x + 10) / 3.0) + 1.0"),
        (True, False, False, "then_else_else", "((x + 10) / 3.0) * 2.0"),
        (False, True, True, "else_then_then", "((-x) - 0.5) + 1.0"),
        (False, True, False, "else_then_else", "((-x) - 0.5) * 2.0"),
        (False, False, True, "else_else_then", "((-x) / 3.0) + 1.0"),
        (False, False, False, "else_else_else", "((-x) / 3.0) * 2.0"),
    ]

    results = {}
    for cond1, cond2, cond3, desc, computation in test_cases:
        inputs = {
            'x': x,
            'cond1': np.array(cond1),
            'cond2': np.array(cond2),
            'cond3': np.array(cond3),
        }
        output = sess.run(None, inputs)[0]
        results[desc] = (output, computation)

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
    print("Test data for nested_if (scoped variables):")
    print("="*80)

    print("\nInput tensor:")
    print(f"Shape: {test_data['input'].shape}")
    print(f"Data: {test_data['input'].flatten().tolist()}")

    for path, (output, computation) in test_data['outputs'].items():
        print(f"\nPath '{path}' - {computation}:")
        print(f"Shape: {output.shape}")
        print(f"Data: {output.flatten().tolist()}")

    print("\n" + "="*80)


if __name__ == '__main__':
    main()
