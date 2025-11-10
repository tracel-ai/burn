#!/usr/bin/env python3
"""
Generate ONNX model with If operator containing Linear (MatMul + Add) operations.
Tests both branches with different linear layer parameters.
"""

import onnx
from onnx import helper, TensorProto, numpy_helper
from onnx.reference import ReferenceEvaluator
import numpy as np


def build_model():
    """Build ONNX model with If operator containing Linear operations in branches."""

    # Input dimensions
    batch_size = 2
    input_size = 5

    # Linear parameters for then branch (input: 5, output: 8)
    then_output_size = 8
    then_weights = np.random.randn(input_size, then_output_size).astype(np.float32) * 0.1
    then_bias = np.random.randn(then_output_size).astype(np.float32) * 0.01

    # Linear parameters for else branch (input: 5, output: 6)
    else_output_size = 6
    else_weights = np.random.randn(input_size, else_output_size).astype(np.float32) * 0.1
    else_bias = np.random.randn(else_output_size).astype(np.float32) * 0.01

    # Create Then branch (Linear + Add constant)
    then_matmul = helper.make_node(
        'MatMul',
        inputs=['x_input', 'then_weights'],
        outputs=['then_matmul_out']
    )

    then_add_bias = helper.make_node(
        'Add',
        inputs=['then_matmul_out', 'then_bias'],
        outputs=['then_linear_out']
    )

    # Add a constant to the linear output
    then_add_const = np.array([0.5], dtype=np.float32)
    then_add = helper.make_node(
        'Add',
        inputs=['then_linear_out', 'then_add_const'],
        outputs=['then_output']
    )

    then_graph = helper.make_graph(
        nodes=[then_matmul, then_add_bias, then_add],
        name='then_branch',
        inputs=[
            helper.make_tensor_value_info('x_input', TensorProto.FLOAT,
                                         [batch_size, input_size])
        ],
        outputs=[
            helper.make_tensor_value_info('then_output', TensorProto.FLOAT,
                                         [batch_size, then_output_size])
        ],
        initializer=[
            numpy_helper.from_array(then_weights, name='then_weights'),
            numpy_helper.from_array(then_bias, name='then_bias'),
            numpy_helper.from_array(then_add_const, name='then_add_const'),
        ]
    )

    # Create Else branch (Linear + Mul constant)
    else_matmul = helper.make_node(
        'MatMul',
        inputs=['x_input', 'else_weights'],
        outputs=['else_matmul_out']
    )

    else_add_bias = helper.make_node(
        'Add',
        inputs=['else_matmul_out', 'else_bias'],
        outputs=['else_linear_out']
    )

    # Multiply by a constant
    else_mul_const = np.array([1.5], dtype=np.float32)
    else_mul = helper.make_node(
        'Mul',
        inputs=['else_linear_out', 'else_mul_const'],
        outputs=['else_output']
    )

    else_graph = helper.make_graph(
        nodes=[else_matmul, else_add_bias, else_mul],
        name='else_branch',
        inputs=[
            helper.make_tensor_value_info('x_input', TensorProto.FLOAT,
                                         [batch_size, input_size])
        ],
        outputs=[
            helper.make_tensor_value_info('else_output', TensorProto.FLOAT,
                                         [batch_size, else_output_size])
        ],
        initializer=[
            numpy_helper.from_array(else_weights, name='else_weights'),
            numpy_helper.from_array(else_bias, name='else_bias'),
            numpy_helper.from_array(else_mul_const, name='else_mul_const'),
        ]
    )

    # Create If node
    if_node = helper.make_node(
        'If',
        inputs=['condition'],
        outputs=['output'],
        then_branch=then_graph,
        else_branch=else_graph,
    )

    # Create main graph
    # Use Identity node to make 'x' available to subgraphs as 'x_input'
    identity = helper.make_node(
        'Identity',
        inputs=['x'],
        outputs=['x_input']
    )

    graph = helper.make_graph(
        nodes=[identity, if_node],
        name='if_linear_model',
        inputs=[
            helper.make_tensor_value_info('x', TensorProto.FLOAT,
                                         [batch_size, input_size]),
            helper.make_tensor_value_info('condition', TensorProto.BOOL, []),
        ],
        outputs=[
            helper.make_tensor_value_info('output', TensorProto.FLOAT,
                                         [batch_size, None])  # Dynamic output size
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

    return model, {
        'then_weights': then_weights,
        'then_bias': then_bias,
        'then_add_const': then_add_const,
        'else_weights': else_weights,
        'else_bias': else_bias,
        'else_mul_const': else_mul_const,
    }


def generate_test_data(model):
    """Generate test inputs and expected outputs using ONNX reference evaluator."""

    batch_size = 2
    input_size = 5

    # Input tensor
    x = np.random.randn(batch_size, input_size).astype(np.float32)

    # Create reference evaluator
    sess = ReferenceEvaluator(model)

    # Then branch computation (condition = True)
    then_output = sess.run(None, {'x': x, 'condition': np.array(True)})[0]

    # Else branch computation (condition = False)
    else_output = sess.run(None, {'x': x, 'condition': np.array(False)})[0]

    return {
        'input': x,
        'condition_true': np.array(True),
        'condition_false': np.array(False),
        'then_output': then_output,
        'else_output': else_output,
    }


def main():
    """Generate model and test data."""

    # Build model
    model, weights = build_model()

    # Save model
    onnx.save(model, 'if_linear.onnx')
    print("✓ Saved if_linear.onnx")

    # Generate test data using ONNX reference implementation
    test_data = generate_test_data(model)

    # Print test data for copying into Rust tests
    print("\n" + "="*80)
    print("Test data for if_linear:")
    print("="*80)

    print("\nInput tensor:")
    print(f"Shape: {test_data['input'].shape}")
    print(f"Data: {test_data['input'].flatten().tolist()}")

    print("\nThen branch output (condition=true):")
    print(f"Shape: {test_data['then_output'].shape}")
    print(f"Data: {test_data['then_output'].flatten().tolist()}")

    print("\nElse branch output (condition=false):")
    print(f"Shape: {test_data['else_output'].shape}")
    print(f"Data: {test_data['else_output'].flatten().tolist()}")

    print("\n" + "="*80)


if __name__ == '__main__':
    main()
