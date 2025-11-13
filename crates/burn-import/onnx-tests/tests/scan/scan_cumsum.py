#!/usr/bin/env python3
"""
Generate ONNX model with Scan operator for cumulative sum.
Tests scan body with Add operation.
"""

import numpy as np
import onnx
from onnx import TensorProto, helper
from onnx.reference import ReferenceEvaluator


def build_model():
    """Build ONNX model with Scan operator for cumulative sum."""

    # Scan body: sum_state = sum_state + input_element
    # Input: sequence of vectors [seq_len, batch, features]
    # Output: cumulative sum at each step

    batch_size = 2
    feature_size = 3
    seq_len = 4

    # Scan body nodes
    # sum_new = sum_state + input_element
    body_add = helper.make_node(
        "Add", inputs=["sum_state_in", "input_element"], outputs=["sum_state_out"]
    )

    # Also output the current sum (scan output)
    body_identity = helper.make_node(
        "Identity", inputs=["sum_state_out"], outputs=["scan_output"]
    )

    # Create scan body graph
    body_graph = helper.make_graph(
        nodes=[body_add, body_identity],
        name="scan_body",
        inputs=[
            # State variables
            helper.make_tensor_value_info(
                "sum_state_in", TensorProto.FLOAT, [batch_size, feature_size]
            ),
            # Scan inputs (one element from sequence)
            helper.make_tensor_value_info(
                "input_element", TensorProto.FLOAT, [batch_size, feature_size]
            ),
        ],
        outputs=[
            # State variables
            helper.make_tensor_value_info(
                "sum_state_out", TensorProto.FLOAT, [batch_size, feature_size]
            ),
            # Scan outputs
            helper.make_tensor_value_info(
                "scan_output", TensorProto.FLOAT, [batch_size, feature_size]
            ),
        ],
    )

    # Create Scan node
    # num_scan_inputs = 1 (the sequence to scan over)
    scan_node = helper.make_node(
        "Scan",
        inputs=["initial_sum", "input_sequence"],
        outputs=["final_sum", "cumsum_sequence"],
        body=body_graph,
        num_scan_inputs=1,
    )

    # Create main graph
    graph = helper.make_graph(
        nodes=[scan_node],
        name="scan_cumsum_model",
        inputs=[
            helper.make_tensor_value_info(
                "initial_sum", TensorProto.FLOAT, [batch_size, feature_size]
            ),
            helper.make_tensor_value_info(
                "input_sequence", TensorProto.FLOAT, [seq_len, batch_size, feature_size]
            ),
        ],
        outputs=[
            helper.make_tensor_value_info(
                "final_sum", TensorProto.FLOAT, [batch_size, feature_size]
            ),
            helper.make_tensor_value_info(
                "cumsum_sequence",
                TensorProto.FLOAT,
                [seq_len, batch_size, feature_size],
            ),
        ],
    )

    # Create model
    model = helper.make_model(
        graph,
        producer_name="burn-import-test",
        opset_imports=[helper.make_opsetid("", 16)],
    )

    # Check model
    onnx.checker.check_model(model)

    return model


def generate_test_data(model):
    """Generate test inputs and expected outputs using ONNX reference evaluator."""

    batch_size = 2
    feature_size = 3
    seq_len = 4

    # Create reference evaluator
    sess = ReferenceEvaluator(model)

    # Test case 1: Random sequence
    initial_sum1 = np.random.randn(batch_size, feature_size).astype(np.float32)
    input_sequence1 = np.random.randn(seq_len, batch_size, feature_size).astype(
        np.float32
    )
    final_sum1, cumsum1 = sess.run(
        None, {"initial_sum": initial_sum1, "input_sequence": input_sequence1}
    )

    test_case1 = {
        "initial_sum": initial_sum1,
        "input_sequence": input_sequence1,
        "final_sum": final_sum1,
        "cumsum_sequence": cumsum1,
        "name": "random",
    }

    # Test case 2: Zeros initial state
    initial_sum2 = np.zeros((batch_size, feature_size), dtype=np.float32)
    input_sequence2 = np.random.randn(seq_len, batch_size, feature_size).astype(
        np.float32
    )
    final_sum2, cumsum2 = sess.run(
        None, {"initial_sum": initial_sum2, "input_sequence": input_sequence2}
    )

    test_case2 = {
        "initial_sum": initial_sum2,
        "input_sequence": input_sequence2,
        "final_sum": final_sum2,
        "cumsum_sequence": cumsum2,
        "name": "zeros_init",
    }

    # Test case 3: Ones sequence
    initial_sum3 = np.random.randn(batch_size, feature_size).astype(np.float32)
    input_sequence3 = np.ones((seq_len, batch_size, feature_size), dtype=np.float32)
    final_sum3, cumsum3 = sess.run(
        None, {"initial_sum": initial_sum3, "input_sequence": input_sequence3}
    )

    test_case3 = {
        "initial_sum": initial_sum3,
        "input_sequence": input_sequence3,
        "final_sum": final_sum3,
        "cumsum_sequence": cumsum3,
        "name": "ones_seq",
    }

    return [test_case1, test_case2, test_case3]


def main():
    """Generate model and test data."""

    # Build model
    model = build_model()

    # Save model
    onnx.save(model, "scan_cumsum.onnx")
    print("✓ Saved scan_cumsum.onnx")

    # Generate test data using ONNX reference implementation
    test_cases = generate_test_data(model)

    # Print test data for copying into Rust tests
    print("\n" + "=" * 80)
    print("Test data for scan_cumsum:")
    print("=" * 80)

    for test_case in test_cases:
        name = test_case["name"]
        print(f"\n--- Test case: {name} ---")

        print(f"\nInitial sum:")
        print(f"Shape: {test_case['initial_sum'].shape}")
        print(f"Data: {test_case['initial_sum'].flatten().tolist()}")

        print(f"\nInput sequence:")
        print(f"Shape: {test_case['input_sequence'].shape}")
        print(f"Data: {test_case['input_sequence'].flatten().tolist()}")

        print(f"\nExpected final sum:")
        print(f"Shape: {test_case['final_sum'].shape}")
        print(f"Data: {test_case['final_sum'].flatten().tolist()}")

        print(f"\nExpected cumsum sequence:")
        print(f"Shape: {test_case['cumsum_sequence'].shape}")
        print(f"Data: {test_case['cumsum_sequence'].flatten().tolist()}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
