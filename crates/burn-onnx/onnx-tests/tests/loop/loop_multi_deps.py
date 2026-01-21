#!/usr/bin/env python3
"""
Generate ONNX model with Loop operator that has multiple loop-carried dependencies.
Tests handling of 3 separate accumulator variables updated independently.
"""

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from onnx.reference import ReferenceEvaluator


def build_model():
    """Build ONNX model with Loop that manages 3 loop-carried dependencies."""

    # Loop body graph
    # Inputs: [iter_num (i64 scalar), condition (bool scalar), accum1, accum2, accum3, x]
    # Outputs: [condition_out, accum1_out, accum2_out, accum3_out]

    # accum1_out = accum1 + x
    add1 = helper.make_node("Add", inputs=["accum1", "x"], outputs=["accum1_out"])

    # accum2_out = accum2 * 2.0
    const_two = np.array([2.0], dtype=np.float32)
    mul2 = helper.make_node(
        "Mul", inputs=["accum2", "two_const"], outputs=["accum2_out"]
    )

    # accum3_out = accum3 - 0.5
    const_half = np.array([0.5], dtype=np.float32)
    sub3 = helper.make_node(
        "Sub", inputs=["accum3", "half_const"], outputs=["accum3_out"]
    )

    # condition_out = condition (always true for this test)
    identity_cond = helper.make_node("Identity", inputs=["cond"], outputs=["cond_out"])

    # ONNX spec requires: each loop-carried dependency must have a corresponding output
    # x is a loop-carried dependency, so we must output it (even if unchanged)
    identity_x = helper.make_node("Identity", inputs=["x"], outputs=["x_out"])

    body_graph = helper.make_graph(
        nodes=[add1, mul2, sub3, identity_cond, identity_x],
        name="loop_body",
        inputs=[
            helper.make_tensor_value_info("iter_num", TensorProto.INT64, []),
            helper.make_tensor_value_info("cond", TensorProto.BOOL, []),
            helper.make_tensor_value_info("accum1", TensorProto.FLOAT, [2, 3]),
            helper.make_tensor_value_info("accum2", TensorProto.FLOAT, [2, 3]),
            helper.make_tensor_value_info("accum3", TensorProto.FLOAT, [2, 3]),
            helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3]),
        ],
        outputs=[
            helper.make_tensor_value_info("cond_out", TensorProto.BOOL, []),
            helper.make_tensor_value_info("accum1_out", TensorProto.FLOAT, [2, 3]),
            helper.make_tensor_value_info("accum2_out", TensorProto.FLOAT, [2, 3]),
            helper.make_tensor_value_info("accum3_out", TensorProto.FLOAT, [2, 3]),
            helper.make_tensor_value_info("x_out", TensorProto.FLOAT, [2, 3]),
        ],
        initializer=[
            numpy_helper.from_array(const_two, name="two_const"),
            numpy_helper.from_array(const_half, name="half_const"),
        ],
    )

    # Create Loop node
    # Per ONNX spec: must have same number of outputs as loop-carried dependencies (4)
    loop_node = helper.make_node(
        "Loop",
        inputs=["M", "cond_init", "accum1_init", "accum2_init", "accum3_init", "x"],
        outputs=["accum1_final", "accum2_final", "accum3_final", "x_final"],
        body=body_graph,
    )

    # Main graph
    graph = helper.make_graph(
        nodes=[loop_node],
        name="loop_multi_deps_model",
        inputs=[
            helper.make_tensor_value_info("M", TensorProto.INT64, []),
            helper.make_tensor_value_info("cond_init", TensorProto.BOOL, []),
            helper.make_tensor_value_info("accum1_init", TensorProto.FLOAT, [2, 3]),
            helper.make_tensor_value_info("accum2_init", TensorProto.FLOAT, [2, 3]),
            helper.make_tensor_value_info("accum3_init", TensorProto.FLOAT, [2, 3]),
            helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3]),
        ],
        outputs=[
            helper.make_tensor_value_info("accum1_final", TensorProto.FLOAT, [2, 3]),
            helper.make_tensor_value_info("accum2_final", TensorProto.FLOAT, [2, 3]),
            helper.make_tensor_value_info("accum3_final", TensorProto.FLOAT, [2, 3]),
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

    # Test with 4 iterations
    M = np.array(4, dtype=np.int64)
    cond_init = np.array(True, dtype=bool)

    # Initial values for 3 accumulators
    accum1_init = np.random.randn(2, 3).astype(np.float32)
    accum2_init = np.random.randn(2, 3).astype(np.float32)
    accum3_init = np.random.randn(2, 3).astype(np.float32)

    # x is not loop-carried, just used in computation
    x = np.random.randn(2, 3).astype(np.float32)

    # Create reference evaluator
    sess = ReferenceEvaluator(model)

    # Run the model
    outputs = sess.run(
        None,
        {
            "M": M,
            "cond_init": cond_init,
            "accum1_init": accum1_init,
            "accum2_init": accum2_init,
            "accum3_init": accum3_init,
            "x": x,
        },
    )

    return {
        "M": M,
        "cond_init": cond_init,
        "accum1_init": accum1_init,
        "accum2_init": accum2_init,
        "accum3_init": accum3_init,
        "x": x,
        "accum1_final": outputs[0],
        "accum2_final": outputs[1],
        "accum3_final": outputs[2],
    }


def main():
    """Generate model and test data."""

    # Build model
    model = build_model()

    # Save model
    onnx.save(model, "loop_multi_deps.onnx")
    print("✓ Saved loop_multi_deps.onnx")

    # Generate test data using ONNX reference implementation
    test_data = generate_test_data(model)

    # Print test data for copying into Rust tests
    print("\n" + "=" * 80)
    print("Test data for loop_multi_deps (4 iterations):")
    print("=" * 80)

    print(f"\nM = {test_data['M']}")
    print(f"cond_init = {test_data['cond_init']}")

    print(f"\naccum1_init shape: {test_data['accum1_init'].shape}")
    print(f"accum1_init data: {test_data['accum1_init'].flatten().tolist()}")

    print(f"\naccum2_init shape: {test_data['accum2_init'].shape}")
    print(f"accum2_init data: {test_data['accum2_init'].flatten().tolist()}")

    print(f"\naccum3_init shape: {test_data['accum3_init'].shape}")
    print(f"accum3_init data: {test_data['accum3_init'].flatten().tolist()}")

    print(f"\nx shape: {test_data['x'].shape}")
    print(f"x data: {test_data['x'].flatten().tolist()}")

    print(f"\naccum1_final shape: {test_data['accum1_final'].shape}")
    print(f"accum1_final data: {test_data['accum1_final'].flatten().tolist()}")

    print(f"\naccum2_final shape: {test_data['accum2_final'].shape}")
    print(f"accum2_final data: {test_data['accum2_final'].flatten().tolist()}")

    print(f"\naccum3_final shape: {test_data['accum3_final'].shape}")
    print(f"accum3_final data: {test_data['accum3_final'].flatten().tolist()}")

    print("\n" + "=" * 80)

    # Verify the logic manually
    print("\nManual verification:")
    print(f"accum1: Each iteration adds x, so final = accum1_init + (4 * x)")
    expected_accum1 = test_data["accum1_init"] + (4 * test_data["x"])
    print(f"Expected accum1: {expected_accum1.flatten().tolist()}")
    print(f"Matches: {np.allclose(expected_accum1, test_data['accum1_final'])}")

    print(f"\naccum2: Each iteration multiplies by 2, so final = accum2_init * (2^4)")
    expected_accum2 = test_data["accum2_init"] * (2**4)
    print(f"Expected accum2: {expected_accum2.flatten().tolist()}")
    print(f"Matches: {np.allclose(expected_accum2, test_data['accum2_final'])}")

    print(f"\naccum3: Each iteration subtracts 0.5, so final = accum3_init - (4 * 0.5)")
    expected_accum3 = test_data["accum3_init"] - (4 * 0.5)
    print(f"Expected accum3: {expected_accum3.flatten().tolist()}")
    print(f"Matches: {np.allclose(expected_accum3, test_data['accum3_final'])}")


if __name__ == "__main__":
    main()
