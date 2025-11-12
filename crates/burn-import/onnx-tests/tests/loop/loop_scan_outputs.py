#!/usr/bin/env python3
"""
Generate ONNX model with Loop operator containing scan outputs.
Tests the full ONNX Loop spec including scan outputs that collect values from each iteration.

Per ONNX spec: https://onnx.ai/onnx/operators/onnx__Loop.html
- Body inputs: [iteration_num, condition, loop_carried_dependencies...]
- Body outputs: [condition, loop_carried_dependencies..., scan_outputs...]
- Scan outputs are concatenated along a new axis (dim 0) to form tensors with shape [num_iterations, ...original_shape]
"""

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from onnx.reference import ReferenceEvaluator


def build_model():
    """Build ONNX model with Loop operator that has scan outputs."""

    # This model implements a loop that:
    # - Accumulates a sum (loop-carried dependency)
    # - Collects intermediate accumulator values (scan output)
    # - Collects iteration numbers (scan output)
    #
    # Loop inputs: [iteration_num, condition, accumulator]
    # Loop outputs: [condition, accumulator_updated, accum_snapshot, iter_snapshot]
    #
    # Main graph outputs:
    # - final_accumulator (final value of loop-carried dep)
    # - accumulated_values (scan output: [num_iters, batch_size, feature_size])
    # - iteration_numbers (scan output: [num_iters])

    batch_size = 2
    feature_size = 3

    # Constant to add each iteration
    add_const = np.array([1.0], dtype=np.float32)

    # Loop body nodes
    # 1. Update accumulator: accumulator = accumulator + 1.0
    body_add = helper.make_node(
        "Add", inputs=["accum_in", "add_const"], outputs=["accum_updated"]
    )

    # 2. Keep looping (always true for fixed iteration count)
    body_identity_cond = helper.make_node(
        "Identity", inputs=["cond_in"], outputs=["cond_out"]
    )

    # 3. Snapshot accumulator for scan output (before update)
    body_snapshot_accum = helper.make_node(
        "Identity", inputs=["accum_in"], outputs=["accum_snapshot"]
    )

    # 4. Snapshot iteration number for scan output
    # Cast to float for scan output (scan outputs must be same type across iterations)
    body_cast_iter = helper.make_node(
        "Cast", inputs=["iter"], outputs=["iter_float"], to=TensorProto.FLOAT
    )

    body_snapshot_iter = helper.make_node(
        "Identity", inputs=["iter_float"], outputs=["iter_snapshot"]
    )

    # Create loop body graph
    # Outputs order: [cond_out, loop_carried_deps..., scan_outputs...]
    body_graph = helper.make_graph(
        nodes=[
            body_add,
            body_identity_cond,
            body_snapshot_accum,
            body_cast_iter,
            body_snapshot_iter,
        ],
        name="loop_body_with_scan",
        inputs=[
            helper.make_tensor_value_info("iter", TensorProto.INT64, []),
            helper.make_tensor_value_info("cond_in", TensorProto.BOOL, []),
            helper.make_tensor_value_info(
                "accum_in", TensorProto.FLOAT, [batch_size, feature_size]
            ),
        ],
        outputs=[
            # Loop-carried outputs (condition + dependencies)
            helper.make_tensor_value_info("cond_out", TensorProto.BOOL, []),
            helper.make_tensor_value_info(
                "accum_updated", TensorProto.FLOAT, [batch_size, feature_size]
            ),
            # Scan outputs (collected from each iteration)
            helper.make_tensor_value_info(
                "accum_snapshot", TensorProto.FLOAT, [batch_size, feature_size]
            ),
            helper.make_tensor_value_info("iter_snapshot", TensorProto.FLOAT, []),
        ],
        initializer=[
            numpy_helper.from_array(add_const, name="add_const"),
        ],
    )

    # Create Loop node
    # Inputs: [max_trip_count, condition, loop_carried_dependencies...]
    # Outputs: [final_loop_carried_deps..., scan_outputs...]
    loop_node = helper.make_node(
        "Loop",
        inputs=["M", "cond", "initial_accum"],
        outputs=["final_accum", "accumulated_values", "iteration_numbers"],
        body=body_graph,
    )

    # Create main graph
    graph = helper.make_graph(
        nodes=[loop_node],
        name="loop_scan_outputs_model",
        inputs=[
            helper.make_tensor_value_info("M", TensorProto.INT64, []),
            helper.make_tensor_value_info("cond", TensorProto.BOOL, []),
            helper.make_tensor_value_info(
                "initial_accum", TensorProto.FLOAT, [batch_size, feature_size]
            ),
        ],
        outputs=[
            # Final value of loop-carried dependency
            helper.make_tensor_value_info(
                "final_accum", TensorProto.FLOAT, [batch_size, feature_size]
            ),
            # Scan output: accumulated values from each iteration
            # Note: Dynamic shape [-1, batch_size, feature_size] where -1 = num_iterations
            helper.make_tensor_value_info(
                "accumulated_values", TensorProto.FLOAT, [None, batch_size, feature_size]
            ),
            # Scan output: iteration numbers from each iteration
            helper.make_tensor_value_info(
                "iteration_numbers", TensorProto.FLOAT, [None]
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

    # Create reference evaluator
    sess = ReferenceEvaluator(model)

    # Test with different iteration counts
    test_cases = []

    # Test case 1: M=3 iterations
    M1 = 3
    initial_accum1 = np.array(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32
    )
    outputs1 = sess.run(
        None,
        {
            "M": np.array(M1, dtype=np.int64),
            "cond": np.array(True),
            "initial_accum": initial_accum1,
        },
    )

    test_cases.append(
        {
            "M": np.array(M1, dtype=np.int64),
            "cond": np.array(True),
            "initial_accum": initial_accum1,
            "final_accum": outputs1[0],
            "accumulated_values": outputs1[1],
            "iteration_numbers": outputs1[2],
            "name": "loop_3_iters",
        }
    )

    # Test case 2: M=5 iterations
    M2 = 5
    initial_accum2 = np.array(
        [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], dtype=np.float32
    )
    outputs2 = sess.run(
        None,
        {
            "M": np.array(M2, dtype=np.int64),
            "cond": np.array(True),
            "initial_accum": initial_accum2,
        },
    )

    test_cases.append(
        {
            "M": np.array(M2, dtype=np.int64),
            "cond": np.array(True),
            "initial_accum": initial_accum2,
            "final_accum": outputs2[0],
            "accumulated_values": outputs2[1],
            "iteration_numbers": outputs2[2],
            "name": "loop_5_iters",
        }
    )

    # Test case 3: M=1 iteration (edge case)
    M3 = 1
    initial_accum3 = np.array(
        [[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]], dtype=np.float32
    )
    outputs3 = sess.run(
        None,
        {
            "M": np.array(M3, dtype=np.int64),
            "cond": np.array(True),
            "initial_accum": initial_accum3,
        },
    )

    test_cases.append(
        {
            "M": np.array(M3, dtype=np.int64),
            "cond": np.array(True),
            "initial_accum": initial_accum3,
            "final_accum": outputs3[0],
            "accumulated_values": outputs3[1],
            "iteration_numbers": outputs3[2],
            "name": "loop_1_iter",
        }
    )

    return test_cases


def main():
    """Generate model and test data."""

    # Build model
    model = build_model()

    # Save model
    onnx.save(model, "loop_scan_outputs.onnx")
    print("✓ Saved loop_scan_outputs.onnx")

    # Generate test data using ONNX reference implementation
    test_cases = generate_test_data(model)

    # Print test data for copying into Rust tests
    print("\n" + "=" * 80)
    print("Test data for loop_scan_outputs:")
    print("=" * 80)

    for test_case in test_cases:
        name = test_case["name"]
        print(f"\n--- Test case: {name} ---")
        print(f"M: {test_case['M']}")

        print(f"\nInitial accumulator:")
        print(f"Shape: {test_case['initial_accum'].shape}")
        print(f"Data: {test_case['initial_accum'].flatten().tolist()}")

        print(f"\nExpected final_accum:")
        print(f"Shape: {test_case['final_accum'].shape}")
        print(f"Data: {test_case['final_accum'].flatten().tolist()}")

        print(f"\nExpected accumulated_values (scan output):")
        print(f"Shape: {test_case['accumulated_values'].shape}")
        print(f"Data: {test_case['accumulated_values'].flatten().tolist()}")

        print(f"\nExpected iteration_numbers (scan output):")
        print(f"Shape: {test_case['iteration_numbers'].shape}")
        print(f"Data: {test_case['iteration_numbers'].flatten().tolist()}")

    print("\n" + "=" * 80)
    print("\nNote: Scan outputs have shape [num_iterations, ...original_shape]")
    print("- accumulated_values: [M, 2, 3]")
    print("- iteration_numbers: [M]")
    print("=" * 80)


if __name__ == "__main__":
    main()
