#!/usr/bin/env python3
"""
Generate ONNX model with Loop operator that terminates dynamically via condition.
Tests early termination when loop body returns condition=False.
"""

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from onnx.reference import ReferenceEvaluator


def build_model():
    """Build ONNX model with Loop that counts down and stops when counter reaches 0."""

    # Loop body graph
    # Inputs: [iter_num (i64 scalar), condition (bool scalar), counter (1D tensor)]
    # Outputs: [condition_out (bool scalar), counter_out (1D tensor)]

    # counter_out = counter - 1
    one_const = np.array([1.0], dtype=np.float32)
    sub_node = helper.make_node(
        "Sub", inputs=["counter", "one_const"], outputs=["counter_out"]
    )

    # Use ReduceMin to convert tensor result to scalar bool
    # First check if counter_out > 0 (returns bool tensor)
    zero_const = np.array([0.0], dtype=np.float32)
    greater_node = helper.make_node(
        "Greater", inputs=["counter_out", "zero_const"], outputs=["greater_out"]
    )

    # Then reduce to scalar (all elements must be > 0)
    reduce_node = helper.make_node(
        "ReduceMin",
        inputs=["greater_out"],
        outputs=["condition_out"],
        keepdims=0,  # Output is scalar
    )

    body_graph = helper.make_graph(
        nodes=[sub_node, greater_node, reduce_node],
        name="loop_body",
        inputs=[
            helper.make_tensor_value_info("iter_num", TensorProto.INT64, []),
            helper.make_tensor_value_info("cond", TensorProto.BOOL, []),
            helper.make_tensor_value_info("counter", TensorProto.FLOAT, [1]),
        ],
        outputs=[
            helper.make_tensor_value_info("condition_out", TensorProto.BOOL, []),
            helper.make_tensor_value_info("counter_out", TensorProto.FLOAT, [1]),
        ],
        initializer=[
            numpy_helper.from_array(one_const, name="one_const"),
            numpy_helper.from_array(zero_const, name="zero_const"),
        ],
    )

    # Create Loop node
    # M=100 (high max iterations), but should stop early when counter reaches 0
    loop_node = helper.make_node(
        "Loop",
        inputs=["M", "cond_init", "counter_init"],
        outputs=["counter_final"],
        body=body_graph,
    )

    # Main graph
    graph = helper.make_graph(
        nodes=[loop_node],
        name="loop_dynamic_cond_model",
        inputs=[
            helper.make_tensor_value_info("M", TensorProto.INT64, []),
            helper.make_tensor_value_info("cond_init", TensorProto.BOOL, []),
            helper.make_tensor_value_info("counter_init", TensorProto.FLOAT, [1]),
        ],
        outputs=[
            helper.make_tensor_value_info("counter_final", TensorProto.FLOAT, [1]),
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

    # Test 1: Counter starts at 5, should run 5 iterations despite M=100
    M = np.array(100, dtype=np.int64)
    cond_init = np.array(True, dtype=bool)

    # Counter (1D tensor with single element) = 5.0
    counter_init = np.array([5.0], dtype=np.float32)

    # Create reference evaluator
    sess = ReferenceEvaluator(model)

    # Run the model
    outputs = sess.run(
        None,
        {
            "M": M,
            "cond_init": cond_init,
            "counter_init": counter_init,
        },
    )

    test1 = {
        "M": M,
        "cond_init": cond_init,
        "counter_init": counter_init,
        "counter_final": outputs[0],
    }

    # Test 2: Counter starts at 3
    counter_init2 = np.array([3.0], dtype=np.float32)
    outputs2 = sess.run(
        None,
        {
            "M": M,
            "cond_init": cond_init,
            "counter_init": counter_init2,
        },
    )

    test2 = {
        "M": M,
        "cond_init": cond_init,
        "counter_init": counter_init2,
        "counter_final": outputs2[0],
    }

    return {"test1": test1, "test2": test2}


def main():
    """Generate model and test data."""

    # Build model
    model = build_model()

    # Save model
    onnx.save(model, "loop_dynamic_cond.onnx")
    print("✓ Saved loop_dynamic_cond.onnx")

    # Generate test data using ONNX reference implementation
    test_data = generate_test_data(model)

    # Print test data for copying into Rust tests
    print("\n" + "=" * 80)
    print("Test data for loop_dynamic_cond:")
    print("=" * 80)

    print("\n--- Test 1: Counter starts at 5.0 ---")
    print(f"M = {test_data['test1']['M']} (max iterations)")
    print(f"cond_init = {test_data['test1']['cond_init']}")
    print(f"counter_init = {test_data['test1']['counter_init']}")
    print(f"counter_final = {test_data['test1']['counter_final']}")
    print(f"Expected: 0.0 (stopped after 5 iterations, not 100)")

    print("\n--- Test 2: Counter starts at 3.0 ---")
    print(f"counter_init = {test_data['test2']['counter_init']}")
    print(f"counter_final = {test_data['test2']['counter_final']}")
    print(f"Expected: 0.0 (stopped after 3 iterations)")

    print("\n" + "=" * 80)

    # Explain the behavior
    print("\nHow it works:")
    print("- Loop body: counter_out = counter - 1, condition_out = (counter_out > 0)")
    print("- Loop continues while condition_out is True")
    print("- With counter_init = 5.0: iterations 1->4, 2->3, 3->2, 4->1, 5->0")
    print("- At iteration 5: counter_out = 0, condition = False, loop stops")
    print("- Result: counter_final = 0 (not negative)")
    print("\nThis tests dynamic termination - loop stops early based on condition,")


if __name__ == "__main__":
    main()
