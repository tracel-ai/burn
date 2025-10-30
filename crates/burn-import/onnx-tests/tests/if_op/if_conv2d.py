#!/usr/bin/env python3
"""
Generate ONNX model with If operator containing Conv2d operations.
Tests both branches with different convolution parameters.
"""

import onnx
from onnx import helper, TensorProto, numpy_helper
from onnx.reference import ReferenceEvaluator
import numpy as np


def build_model():
    """Build ONNX model with If operator containing Conv2d in branches."""

    # Input dimensions: [batch, channels, height, width]
    # Using small dimensions for testing
    batch_size = 1
    in_channels = 2
    height = 4
    width = 4

    # Conv parameters for then branch (3x3 kernel, 4 output channels)
    then_out_channels = 4
    then_kernel_size = 3
    then_conv_weights = np.random.randn(
        then_out_channels, in_channels, then_kernel_size, then_kernel_size
    ).astype(np.float32) * 0.1
    then_conv_bias = np.random.randn(then_out_channels).astype(np.float32) * 0.01

    # Conv parameters for else branch (1x1 kernel, 3 output channels)
    else_out_channels = 3
    else_kernel_size = 1
    else_conv_weights = np.random.randn(
        else_out_channels, in_channels, else_kernel_size, else_kernel_size
    ).astype(np.float32) * 0.1
    else_conv_bias = np.random.randn(else_out_channels).astype(np.float32) * 0.01

    # Create Then branch (Conv2d 3x3 + constant Add)
    then_conv = helper.make_node(
        'Conv',
        inputs=['x_input', 'then_conv_weights', 'then_conv_bias'],
        outputs=['then_conv_out'],
        kernel_shape=[then_kernel_size, then_kernel_size],
        pads=[1, 1, 1, 1],  # SAME padding
    )

    # Add a constant to the conv output
    then_add_const = np.array([1.0], dtype=np.float32)
    then_add = helper.make_node(
        'Add',
        inputs=['then_conv_out', 'then_add_const'],
        outputs=['then_output']
    )

    then_graph = helper.make_graph(
        nodes=[then_conv, then_add],
        name='then_branch',
        inputs=[
            helper.make_tensor_value_info('x_input', TensorProto.FLOAT,
                                         [batch_size, in_channels, height, width])
        ],
        outputs=[
            helper.make_tensor_value_info('then_output', TensorProto.FLOAT,
                                         [batch_size, then_out_channels, height, width])
        ],
        initializer=[
            numpy_helper.from_array(then_conv_weights, name='then_conv_weights'),
            numpy_helper.from_array(then_conv_bias, name='then_conv_bias'),
            numpy_helper.from_array(then_add_const, name='then_add_const'),
        ]
    )

    # Create Else branch (Conv2d 1x1 + constant Mul)
    else_conv = helper.make_node(
        'Conv',
        inputs=['x_input', 'else_conv_weights', 'else_conv_bias'],
        outputs=['else_conv_out'],
        kernel_shape=[else_kernel_size, else_kernel_size],
    )

    # Multiply by a constant
    else_mul_const = np.array([2.0], dtype=np.float32)
    else_mul = helper.make_node(
        'Mul',
        inputs=['else_conv_out', 'else_mul_const'],
        outputs=['else_output']
    )

    else_graph = helper.make_graph(
        nodes=[else_conv, else_mul],
        name='else_branch',
        inputs=[
            helper.make_tensor_value_info('x_input', TensorProto.FLOAT,
                                         [batch_size, in_channels, height, width])
        ],
        outputs=[
            helper.make_tensor_value_info('else_output', TensorProto.FLOAT,
                                         [batch_size, else_out_channels, height, width])
        ],
        initializer=[
            numpy_helper.from_array(else_conv_weights, name='else_conv_weights'),
            numpy_helper.from_array(else_conv_bias, name='else_conv_bias'),
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
    # Note: In ONNX, the main graph input 'x' is implicitly available to subgraphs as 'x_input'
    # We need to use an Identity node to make 'x' available to the If node's subgraphs
    identity = helper.make_node(
        'Identity',
        inputs=['x'],
        outputs=['x_input']
    )

    graph = helper.make_graph(
        nodes=[identity, if_node],
        name='if_conv2d_model',
        inputs=[
            helper.make_tensor_value_info('x', TensorProto.FLOAT,
                                         [batch_size, in_channels, height, width]),
            helper.make_tensor_value_info('condition', TensorProto.BOOL, []),
        ],
        outputs=[
            helper.make_tensor_value_info('output', TensorProto.FLOAT,
                                         [batch_size, None, height, width])  # Dynamic channels
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
        'then_conv_weights': then_conv_weights,
        'then_conv_bias': then_conv_bias,
        'then_add_const': then_add_const,
        'else_conv_weights': else_conv_weights,
        'else_conv_bias': else_conv_bias,
        'else_mul_const': else_mul_const,
    }


def generate_test_data(model):
    """Generate test inputs and expected outputs using ONNX reference evaluator."""

    batch_size = 1
    in_channels = 2
    height = 4
    width = 4

    # Input tensor
    x = np.random.randn(batch_size, in_channels, height, width).astype(np.float32)

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
    onnx.save(model, 'if_conv2d.onnx')
    print("✓ Saved if_conv2d.onnx")

    # Generate test data using ONNX reference implementation
    test_data = generate_test_data(model)

    # Print test data for copying into Rust tests
    print("\n" + "="*80)
    print("Test data for if_conv2d:")
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
# timestamp
