#!/usr/bin/env python3
"""
Generate ONNX model with Scan operator that scans in reverse direction.
Tests scan_input_directions attribute handling.
"""

import numpy as np
import onnx
from onnx import TensorProto, helper
from onnx.reference import ReferenceEvaluator


def build_model():
    """Build ONNX model with Scan that processes sequence in reverse."""

    # Scan body graph - cumulative sum
    # Inputs: [state (accumulated sum), scan_input (current element)]
    # Outputs: [state_out (updated sum), scan_out (current sum)]

    add_node = helper.make_node("Add", inputs=["sum_in", "x"], outputs=["sum_out"])

    # Scan output is the current accumulated sum
    identity = helper.make_node("Identity", inputs=["sum_out"], outputs=["scan_out"])

    body_graph = helper.make_graph(
        nodes=[add_node, identity],
        name="scan_body",
        inputs=[
            helper.make_tensor_value_info("sum_in", TensorProto.FLOAT, [2, 3]),
            helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3]),
        ],
        outputs=[
            helper.make_tensor_value_info("sum_out", TensorProto.FLOAT, [2, 3]),
            helper.make_tensor_value_info("scan_out", TensorProto.FLOAT, [2, 3]),
        ],
    )

    # Create Scan node with reverse direction
    scan_node = helper.make_node(
        "Scan",
        inputs=["initial_sum", "input_sequence"],
        outputs=["final_sum", "cumsum_sequence"],
        num_scan_inputs=1,
        body=body_graph,
        scan_input_directions=[1],  # 1 = reverse (scan from end to start)
        scan_output_directions=[1],  # Output in reverse order too
    )

    # Main graph
    graph = helper.make_graph(
        nodes=[scan_node],
        name="scan_reverse_model",
        inputs=[
            helper.make_tensor_value_info("initial_sum", TensorProto.FLOAT, [2, 3]),
            helper.make_tensor_value_info(
                "input_sequence", TensorProto.FLOAT, [4, 2, 3]
            ),
        ],
        outputs=[
            helper.make_tensor_value_info("final_sum", TensorProto.FLOAT, [2, 3]),
            helper.make_tensor_value_info(
                "cumsum_sequence", TensorProto.FLOAT, [4, 2, 3]
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


def generate_test_data(model):
    """Generate test inputs and expected outputs using ONNX reference evaluator."""

    # Initial sum state
    initial_sum = np.random.randn(2, 3).astype(np.float32)

    # Input sequence (4 timesteps)
    input_sequence = np.random.randn(4, 2, 3).astype(np.float32)

    # Create reference evaluator
    sess = ReferenceEvaluator(model)

    # Run the model
    outputs = sess.run(
        None,
        {
            "initial_sum": initial_sum,
            "input_sequence": input_sequence,
        },
    )

    return {
        "initial_sum": initial_sum,
        "input_sequence": input_sequence,
        "final_sum": outputs[0],
        "cumsum_sequence": outputs[1],
    }


def main():
    """Generate model and test data."""

    # Build model
    model = build_model()

    # Save model
    onnx.save(model, "scan_reverse.onnx")
    print("✓ Saved scan_reverse.onnx")

    # Generate test data using ONNX reference implementation
    test_data = generate_test_data(model)

    # Print test data for copying into Rust tests
    print("\n" + "=" * 80)
    print("Test data for scan_reverse (reverse cumsum):")
    print("=" * 80)

    print(f"\ninitial_sum shape: {test_data['initial_sum'].shape}")
    print(f"initial_sum data: {test_data['initial_sum'].flatten().tolist()}")

    print(f"\ninput_sequence shape: {test_data['input_sequence'].shape}")
    print(f"input_sequence data: {test_data['input_sequence'].flatten().tolist()}")

    print(f"\nfinal_sum shape: {test_data['final_sum'].shape}")
    print(f"final_sum data: {test_data['final_sum'].flatten().tolist()}")

    print(f"\ncumsum_sequence shape: {test_data['cumsum_sequence'].shape}")
    print(f"cumsum_sequence data: {test_data['cumsum_sequence'].flatten().tolist()}")

    print("\n" + "=" * 80)

    # Verify reverse cumsum logic
    print("\nManual verification (reverse cumsum):")
    print("Input sequence (timesteps 0-3):")
    for i in range(4):
        print(f"  timestep {i}: {test_data['input_sequence'][i].flatten().tolist()}")

    # Reverse cumsum: Start from end, accumulate backward
    # Process order: 3, 2, 1, 0
    # Output order (with scan_output_directions=[1]): also reversed
    print("\nReverse processing order: 3 -> 2 -> 1 -> 0")

    manual_sum = test_data["initial_sum"].copy()
    reverse_outputs = []
    for i in range(3, -1, -1):  # Process in reverse: 3, 2, 1, 0
        manual_sum = manual_sum + test_data["input_sequence"][i]
        reverse_outputs.append(manual_sum.copy())

    # Since scan_output_directions=[1], outputs are also in reverse
    # So reverse_outputs[0] corresponds to cumsum_sequence[0]
    print(f"\nExpected final_sum (sum of all): {manual_sum.flatten().tolist()}")
    print(f"Actual final_sum: {test_data['final_sum'].flatten().tolist()}")
    print(f"Matches: {np.allclose(manual_sum, test_data['final_sum'])}")


if __name__ == "__main__":
    main()
