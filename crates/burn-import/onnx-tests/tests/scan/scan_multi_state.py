#!/usr/bin/env python3
"""
Generate ONNX model with Scan operator that has multiple state variables.
Tests LSTM-like pattern with 2 state variables (hidden state + cell state).
"""

import numpy as np
import onnx
from onnx import TensorProto, helper
from onnx.reference import ReferenceEvaluator


def build_model():
    """Build ONNX model with Scan that manages 2 state variables like LSTM."""

    # Scan body graph - simplified LSTM-like update
    # IMPORTANT: Body inputs order must be: [state_vars..., scan_inputs...]
    # Inputs: [hidden_state, cell_state, input_element]
    # Outputs: [hidden_state_out, cell_state_out, output_element]

    # Simple update rule (independent updates to avoid dependency issues):
    # hidden_out = hidden + input
    # cell_out = cell + input
    # output = hidden_out

    # Note: Need to reference the correct variable names - they come from the body graph inputs
    add_hidden = helper.make_node(
        "Add", inputs=["hidden_state", "input_elem"], outputs=["hidden_out"]
    )

    add_cell = helper.make_node(
        "Add", inputs=["cell_state", "input_elem"], outputs=["cell_out"]
    )

    # Output is just the hidden state
    identity = helper.make_node("Identity", inputs=["hidden_out"], outputs=["scan_out"])

    body_graph = helper.make_graph(
        nodes=[add_hidden, add_cell, identity],
        name="scan_body",
        inputs=[
            # State variables come first
            helper.make_tensor_value_info("hidden_state", TensorProto.FLOAT, [2, 3]),
            helper.make_tensor_value_info("cell_state", TensorProto.FLOAT, [2, 3]),
            # Then scan inputs
            helper.make_tensor_value_info("input_elem", TensorProto.FLOAT, [2, 3]),
        ],
        outputs=[
            # Updated state variables (same order as inputs)
            helper.make_tensor_value_info("hidden_out", TensorProto.FLOAT, [2, 3]),
            helper.make_tensor_value_info("cell_out", TensorProto.FLOAT, [2, 3]),
            # Scan outputs
            helper.make_tensor_value_info("scan_out", TensorProto.FLOAT, [2, 3]),
        ],
    )

    # Create Scan node with 2 state variables
    scan_node = helper.make_node(
        "Scan",
        inputs=["initial_hidden", "initial_cell", "input_sequence"],
        outputs=["final_hidden", "final_cell", "output_sequence"],
        num_scan_inputs=1,  # Only input_sequence is scan input
        body=body_graph,
    )

    # Main graph
    graph = helper.make_graph(
        nodes=[scan_node],
        name="scan_multi_state_model",
        inputs=[
            helper.make_tensor_value_info("initial_hidden", TensorProto.FLOAT, [2, 3]),
            helper.make_tensor_value_info("initial_cell", TensorProto.FLOAT, [2, 3]),
            helper.make_tensor_value_info(
                "input_sequence", TensorProto.FLOAT, [4, 2, 3]
            ),
        ],
        outputs=[
            helper.make_tensor_value_info("final_hidden", TensorProto.FLOAT, [2, 3]),
            helper.make_tensor_value_info("final_cell", TensorProto.FLOAT, [2, 3]),
            helper.make_tensor_value_info(
                "output_sequence", TensorProto.FLOAT, [4, 2, 3]
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

    # Initial states
    initial_hidden = np.random.randn(2, 3).astype(np.float32)
    initial_cell = np.random.randn(2, 3).astype(np.float32)

    # Input sequence (4 timesteps)
    input_sequence = np.random.randn(4, 2, 3).astype(np.float32)

    # Create reference evaluator
    sess = ReferenceEvaluator(model)

    # Run the model
    outputs = sess.run(
        None,
        {
            "initial_hidden": initial_hidden,
            "initial_cell": initial_cell,
            "input_sequence": input_sequence,
        },
    )

    return {
        "initial_hidden": initial_hidden,
        "initial_cell": initial_cell,
        "input_sequence": input_sequence,
        "final_hidden": outputs[0],
        "final_cell": outputs[1],
        "output_sequence": outputs[2],
    }


def main():
    """Generate model and test data."""

    # Build model
    model = build_model()

    # Save model
    onnx.save(model, "scan_multi_state.onnx")
    print("✓ Saved scan_multi_state.onnx")

    # Generate test data using ONNX reference implementation
    test_data = generate_test_data(model)

    # Print test data for copying into Rust tests
    print("\n" + "=" * 80)
    print("Test data for scan_multi_state (2 state variables):")
    print("=" * 80)

    print(f"\ninitial_hidden shape: {test_data['initial_hidden'].shape}")
    print(f"initial_hidden data: {test_data['initial_hidden'].flatten().tolist()}")

    print(f"\ninitial_cell shape: {test_data['initial_cell'].shape}")
    print(f"initial_cell data: {test_data['initial_cell'].flatten().tolist()}")

    print(f"\ninput_sequence shape: {test_data['input_sequence'].shape}")
    print(f"input_sequence data: {test_data['input_sequence'].flatten().tolist()}")

    print(f"\nfinal_hidden shape: {test_data['final_hidden'].shape}")
    print(f"final_hidden data: {test_data['final_hidden'].flatten().tolist()}")

    print(f"\nfinal_cell shape: {test_data['final_cell'].shape}")
    print(f"final_cell data: {test_data['final_cell'].flatten().tolist()}")

    print(f"\noutput_sequence shape: {test_data['output_sequence'].shape}")
    print(f"output_sequence data: {test_data['output_sequence'].flatten().tolist()}")

    print("\n" + "=" * 80)

    # Verify the logic manually
    print("\nManual verification:")
    print("Update rule: hidden_out = hidden + input, cell_out = cell + input")

    hidden = test_data["initial_hidden"].copy()
    cell = test_data["initial_cell"].copy()

    for i in range(4):
        hidden = hidden + test_data["input_sequence"][i]
        cell = cell + test_data["input_sequence"][i]
        print(
            f"After timestep {i}: hidden sum = {hidden.sum():.4f}, cell sum = {cell.sum():.4f}"
        )

    print(f"\nExpected final_hidden sum: {hidden.sum():.4f}")
    print(f"Actual final_hidden sum: {test_data['final_hidden'].sum():.4f}")
    print(f"Matches: {np.allclose(hidden, test_data['final_hidden'])}")

    print(f"\nExpected final_cell sum: {cell.sum():.4f}")
    print(f"Actual final_cell sum: {test_data['final_cell'].sum():.4f}")
    print(f"Matches: {np.allclose(cell, test_data['final_cell'])}")


if __name__ == "__main__":
    main()
