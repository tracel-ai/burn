#!/usr/bin/env python3
"""
Test case: Reshape(scalar, [-1]) -> Scalar

This tests the optimization where reshaping a scalar with shape [-1] or [1]
keeps the output as a scalar instead of converting to a rank-1 tensor.

This pattern appears in models like Silero VAD where scalar values are
reshaped but should remain scalars for efficiency.
"""

import numpy as np
import onnx
from onnx import helper, TensorProto


def main():
    # Create a model that:
    # 1. Takes a 1x1 tensor input
    # 2. Reshapes it to scalar (empty shape)
    # 3. Reshapes the scalar to [-1] (should remain scalar)
    # 4. Returns the result

    # Input: 1x1 tensor
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1])

    # Output: scalar (should remain scalar after Reshape with [-1])
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [])

    # Shape tensor for first reshape: empty shape = scalar
    shape_to_scalar = helper.make_tensor(
        name="shape_to_scalar",
        data_type=TensorProto.INT64,
        dims=[0],
        vals=[]
    )

    # Shape tensor for second reshape: [-1]
    shape_neg1 = helper.make_tensor(
        name="shape_neg1",
        data_type=TensorProto.INT64,
        dims=[1],
        vals=[-1]
    )

    # First Reshape: [1,1] tensor -> scalar
    reshape1 = helper.make_node(
        "Reshape",
        inputs=["input", "shape_to_scalar"],
        outputs=["scalar_val"],
        name="reshape_to_scalar"
    )

    # Second Reshape: scalar -> [-1] (should remain scalar with optimization)
    reshape2 = helper.make_node(
        "Reshape",
        inputs=["scalar_val", "shape_neg1"],
        outputs=["output"],
        name="reshape_scalar_neg1"
    )

    # Create the graph
    graph_def = helper.make_graph(
        [reshape1, reshape2],
        "reshape_scalar_to_scalar_model",
        [input_tensor],
        [output_tensor],
        [shape_to_scalar, shape_neg1]
    )

    # Create the model
    model_def = helper.make_model(
        graph_def,
        producer_name="reshape_scalar_to_scalar",
        opset_imports=[helper.make_operatorsetid("", 16)]
    )

    # Save the model
    onnx.save(model_def, "reshape_scalar_to_scalar.onnx")
    print("Model exported successfully to reshape_scalar_to_scalar.onnx")
    print("Model structure: Reshape([1,1] -> scalar) -> Reshape(scalar, [-1]) -> scalar")

    # Verify with ONNX Runtime or reference evaluator
    try:
        from onnx.reference import ReferenceEvaluator

        test_input = np.array([[42.5]], dtype=np.float32)
        print(f"Test input shape: {test_input.shape}")
        print(f"Test input value: {test_input}")

        sess = ReferenceEvaluator(model_def)
        result = sess.run(None, {"input": test_input})

        print(f"ONNX output shape: {result[0].shape}")
        print(f"ONNX output value: {result[0]}")
        print(f"ONNX output dtype: {result[0].dtype}")

    except ImportError:
        print("onnx.reference not available, skipping verification")


if __name__ == "__main__":
    main()
