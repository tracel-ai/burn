#!/usr/bin/env python3
"""
Generate ONNX model with Loop operator containing simple operations.
Tests loop body with Add and Mul operations.
"""

import onnx
from onnx import helper, TensorProto, numpy_helper
from onnx.reference import ReferenceEvaluator
import numpy as np


def build_model():
    """Build ONNX model with Loop operator."""

    # Loop body: accumulator = accumulator + input + 1.0, then multiply by 2.0
    # Loop iteration count: M
    # Loop inputs: [iteration_num, condition, accumulator, x]
    # Loop outputs: [condition, accumulator]

    batch_size = 2
    feature_size = 3

    # Constants for loop body
    add_const = np.array([1.0], dtype=np.float32)
    mul_const = np.array([2.0], dtype=np.float32)

    # Loop body nodes
    # accumulator_new = accumulator + x + 1.0
    body_add1 = helper.make_node(
        'Add',
        inputs=['accum_in', 'x_in'],
        outputs=['add1_out']
    )

    body_add2 = helper.make_node(
        'Add',
        inputs=['add1_out', 'add_const'],
        outputs=['add2_out']
    )

    # accumulator_new = accumulator_new * 2.0
    body_mul = helper.make_node(
        'Mul',
        inputs=['add2_out', 'mul_const'],
        outputs=['accum_out']
    )

    # Keep looping (always true for fixed iteration count)
    body_identity_cond = helper.make_node(
        'Identity',
        inputs=['cond_in'],
        outputs=['cond_out']
    )

    # ONNX spec requires: each loop-carried dependency must have a corresponding output
    # x is a loop-carried dependency, so we must output it (even if unchanged)
    body_identity_x = helper.make_node(
        'Identity',
        inputs=['x_in'],
        outputs=['x_out']
    )

    # Create loop body graph
    body_graph = helper.make_graph(
        nodes=[body_add1, body_add2, body_mul, body_identity_cond, body_identity_x],
        name='loop_body',
        inputs=[
            helper.make_tensor_value_info('iter', TensorProto.INT64, []),
            helper.make_tensor_value_info('cond_in', TensorProto.BOOL, []),
            helper.make_tensor_value_info('accum_in', TensorProto.FLOAT,
                                         [batch_size, feature_size]),
            helper.make_tensor_value_info('x_in', TensorProto.FLOAT,
                                         [batch_size, feature_size]),
        ],
        outputs=[
            helper.make_tensor_value_info('cond_out', TensorProto.BOOL, []),
            helper.make_tensor_value_info('accum_out', TensorProto.FLOAT,
                                         [batch_size, feature_size]),
            helper.make_tensor_value_info('x_out', TensorProto.FLOAT,
                                         [batch_size, feature_size]),
        ],
        initializer=[
            numpy_helper.from_array(add_const, name='add_const'),
            numpy_helper.from_array(mul_const, name='mul_const'),
        ]
    )

    # Create Loop node
    # Inputs: [max_trip_count, condition, loop_carried_dependencies...]
    # Outputs: [final values of loop-carried dependencies]
    # Per ONNX spec: must have same number of outputs as loop-carried dependencies
    loop_node = helper.make_node(
        'Loop',
        inputs=['M', 'cond', 'initial_accum', 'x'],
        outputs=['final_accum', 'x_final'],
        body=body_graph,
    )

    # Create main graph
    graph = helper.make_graph(
        nodes=[loop_node],
        name='loop_simple_model',
        inputs=[
            helper.make_tensor_value_info('M', TensorProto.INT64, []),
            helper.make_tensor_value_info('cond', TensorProto.BOOL, []),
            helper.make_tensor_value_info('initial_accum', TensorProto.FLOAT,
                                         [batch_size, feature_size]),
            helper.make_tensor_value_info('x', TensorProto.FLOAT,
                                         [batch_size, feature_size]),
        ],
        outputs=[
            helper.make_tensor_value_info('final_accum', TensorProto.FLOAT,
                                         [batch_size, feature_size])
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
    feature_size = 3

    # Create reference evaluator
    sess = ReferenceEvaluator(model)

    # Test with different iteration counts
    test_cases = []

    # Test case 1: M=3
    M1 = 3
    initial_accum1 = np.random.randn(batch_size, feature_size).astype(np.float32)
    x1 = np.random.randn(batch_size, feature_size).astype(np.float32)
    output1 = sess.run(None, {
        'M': np.array(M1, dtype=np.int64),
        'cond': np.array(True),
        'initial_accum': initial_accum1,
        'x': x1
    })[0]

    test_cases.append({
        'M': np.array(M1, dtype=np.int64),
        'cond': np.array(True),
        'initial_accum': initial_accum1,
        'x': x1,
        'output': output1,
        'name': 'loop_3'
    })

    # Test case 2: M=5
    M2 = 5
    initial_accum2 = np.random.randn(batch_size, feature_size).astype(np.float32)
    x2 = np.random.randn(batch_size, feature_size).astype(np.float32)
    output2 = sess.run(None, {
        'M': np.array(M2, dtype=np.int64),
        'cond': np.array(True),
        'initial_accum': initial_accum2,
        'x': x2
    })[0]

    test_cases.append({
        'M': np.array(M2, dtype=np.int64),
        'cond': np.array(True),
        'initial_accum': initial_accum2,
        'x': x2,
        'output': output2,
        'name': 'loop_5'
    })

    # Test case 3: M=0 (no iterations)
    M3 = 0
    initial_accum3 = np.random.randn(batch_size, feature_size).astype(np.float32)
    x3 = np.random.randn(batch_size, feature_size).astype(np.float32)
    output3 = sess.run(None, {
        'M': np.array(M3, dtype=np.int64),
        'cond': np.array(True),
        'initial_accum': initial_accum3,
        'x': x3
    })[0]

    test_cases.append({
        'M': np.array(M3, dtype=np.int64),
        'cond': np.array(True),
        'initial_accum': initial_accum3,
        'x': x3,
        'output': output3,
        'name': 'loop_0'
    })

    return test_cases


def main():
    """Generate model and test data."""

    # Build model
    model = build_model()

    # Save model
    onnx.save(model, 'loop_simple.onnx')
    print("✓ Saved loop_simple.onnx")

    # Generate test data using ONNX reference implementation
    test_cases = generate_test_data(model)

    # Print test data for copying into Rust tests
    print("\n" + "="*80)
    print("Test data for loop_simple:")
    print("="*80)

    for test_case in test_cases:
        name = test_case['name']
        print(f"\n--- Test case: {name} ---")
        print(f"M: {test_case['M']}")

        print(f"\nInitial accumulator:")
        print(f"Shape: {test_case['initial_accum'].shape}")
        print(f"Data: {test_case['initial_accum'].flatten().tolist()}")

        print(f"\nX tensor:")
        print(f"Shape: {test_case['x'].shape}")
        print(f"Data: {test_case['x'].flatten().tolist()}")

        print(f"\nExpected output:")
        print(f"Shape: {test_case['output'].shape}")
        print(f"Data: {test_case['output'].flatten().tolist()}")

    print("\n" + "="*80)


if __name__ == '__main__':
    main()
