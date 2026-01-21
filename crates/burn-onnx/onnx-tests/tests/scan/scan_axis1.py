#!/usr/bin/env python3
"""
Generate ONNX model with Scan operator using scan_input_axes=[1].
Tests that the scan operator correctly slices along axis 1.

NOTE: ONNX ReferenceEvaluator does not support scan_input_axes != [0].
Error: "Scan is not implemented for other input axes than 0."
Therefore, we manually compute expected outputs instead of using ReferenceEvaluator.
The ONNX spec supports arbitrary scan axes, but the reference implementation is limited.
"""

import numpy as np
import onnx
from onnx import TensorProto, helper


def build_model():
    """Build ONNX model with Scan operator scanning along axis 1."""

    # Input shape: [batch=2, seq_len=3, features=2]
    # scan_input_axes=[1] means scan along axis 1 (sequence dimension)
    batch_size = 2
    seq_len = 3
    feature_size = 2

    # Scan body: sum_state = sum_state + input_element
    body_add = helper.make_node(
        "Add", inputs=["sum_state_in", "input_element"], outputs=["sum_state_out"]
    )

    body_identity = helper.make_node(
        "Identity", inputs=["sum_state_out"], outputs=["scan_output"]
    )

    # Create scan body graph
    # Body receives elements of shape [batch, features] (axis 1 removed)
    body_graph = helper.make_graph(
        nodes=[body_add, body_identity],
        name="scan_body",
        inputs=[
            # State variables
            helper.make_tensor_value_info(
                "sum_state_in", TensorProto.FLOAT, [batch_size, feature_size]
            ),
            # Scan inputs (one element from sequence along axis 1)
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

    # Create Scan node with scan_input_axes=[1] and scan_output_axes=[1]
    scan_node = helper.make_node(
        "Scan",
        inputs=["initial_sum", "input_sequence"],
        outputs=["final_sum", "cumsum_sequence"],
        body=body_graph,
        num_scan_inputs=1,
        scan_input_axes=[1],  # Scan along axis 1 instead of default axis 0
        scan_output_axes=[1],  # Stack outputs along axis 1 as well
    )

    # Create main graph
    graph = helper.make_graph(
        nodes=[scan_node],
        name="scan_axis1_model",
        inputs=[
            helper.make_tensor_value_info(
                "initial_sum", TensorProto.FLOAT, [batch_size, feature_size]
            ),
            helper.make_tensor_value_info(
                "input_sequence", TensorProto.FLOAT, [batch_size, seq_len, feature_size]
            ),
        ],
        outputs=[
            helper.make_tensor_value_info(
                "final_sum", TensorProto.FLOAT, [batch_size, feature_size]
            ),
            helper.make_tensor_value_info(
                "cumsum_sequence",
                TensorProto.FLOAT,
                [batch_size, seq_len, feature_size],
            ),
        ],
    )

    # Create model
    model = helper.make_model(
        graph,
        producer_name="burn-onnx-test",
        opset_imports=[helper.make_opsetid("", 16)],
    )

    # Check model
    onnx.checker.check_model(model)

    return model


def compute_expected_outputs():
    """
    Manually compute expected outputs for scan along axis 1.

    Input shape: [2, 3, 2] (batch=2, seq=3, features=2)
    Scanning along axis 1 (seq dimension)

    For each batch:
      - iteration 0: sum += input[:, 0, :]  (shape [2, 2])
      - iteration 1: sum += input[:, 1, :]  (shape [2, 2])
      - iteration 2: sum += input[:, 2, :]  (shape [2, 2])
    """

    batch_size = 2
    seq_len = 3
    feature_size = 2

    # Initial sum: zeros
    initial_sum = np.zeros((batch_size, feature_size), dtype=np.float32)

    # Input sequence with simple pattern for easy verification
    # Batch 0: [[1, 2], [3, 4], [5, 6]]
    # Batch 1: [[10, 20], [30, 40], [50, 60]]
    input_sequence = np.array([
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        [[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]]
    ], dtype=np.float32)

    # Compute expected outputs by simulating scan
    sum_state = initial_sum.copy()
    cumsum_list = []

    for seq_idx in range(seq_len):
        # Extract element at [:, seq_idx, :] for all batches
        element = input_sequence[:, seq_idx, :]  # shape [2, 2]
        sum_state = sum_state + element
        cumsum_list.append(sum_state.copy())

    final_sum = sum_state
    # Stack along axis 1 to get [batch, seq, features]
    cumsum_sequence = np.stack(cumsum_list, axis=1)

    return initial_sum, input_sequence, final_sum, cumsum_sequence


def main():
    """Generate model and test data."""

    # Build model
    model = build_model()

    # Save model
    onnx.save(model, "scan_axis1.onnx")
    print("✓ Saved scan_axis1.onnx")

    # Manually compute expected outputs
    initial_sum, input_sequence, final_sum, cumsum_sequence = compute_expected_outputs()

    # Print test data for copying into Rust tests
    print("\n" + "=" * 80)
    print("Test data for scan_axis1 (scan along axis 1):")
    print("=" * 80)
    print("\nInitial sum:")
    print(f"Shape: {initial_sum.shape}")
    print(f"Data: {initial_sum.flatten().tolist()}")

    print("\nInput sequence:")
    print(f"Shape: {input_sequence.shape}")
    print(f"Data: {input_sequence.flatten().tolist()}")
    print(f"Batch 0: {input_sequence[0].tolist()}")
    print(f"Batch 1: {input_sequence[1].tolist()}")

    print("\nExpected final sum:")
    print(f"Shape: {final_sum.shape}")
    print(f"Data: {final_sum.flatten().tolist()}")
    print(f"Batch 0: {final_sum[0].tolist()} = [0,0] + [1,2] + [3,4] + [5,6]")
    print(f"Batch 1: {final_sum[1].tolist()} = [0,0] + [10,20] + [30,40] + [50,60]")

    print("\nExpected cumsum sequence:")
    print(f"Shape: {cumsum_sequence.shape}")
    print(f"Data: {cumsum_sequence.flatten().tolist()}")
    print(f"Batch 0: {cumsum_sequence[0].tolist()}")
    print(f"Batch 1: {cumsum_sequence[1].tolist()}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
