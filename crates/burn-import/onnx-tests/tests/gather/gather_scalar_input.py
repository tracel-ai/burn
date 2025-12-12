#!/usr/bin/env python3
"""
Test case: Gather(scalar, index) -> Scalar

This tests gathering from a scalar input. When the input is a scalar,
there's only one element to gather, so the output is always that scalar.

This pattern appears when Reshape(scalar, [-1]) is followed by Gather,
which happens in models like Silero VAD.
"""

import numpy as np
import onnx
from onnx import helper, TensorProto


def main():
    # Create a model that:
    # 1. Takes a 1x1 tensor input
    # 2. Reshapes it to scalar
    # 3. Reshapes the scalar with shape [1] (should remain scalar with optimization)
    # 4. Gathers from the scalar (should pass through the scalar)
    # 5. Returns the result

    # Input: 1x1 tensor
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1])

    # Output: scalar
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [])

    # Shape tensor for first reshape: empty shape = scalar
    shape_to_scalar = helper.make_tensor(
        name="shape_to_scalar",
        data_type=TensorProto.INT64,
        dims=[0],
        vals=[]
    )

    # Shape tensor for second reshape: [1]
    shape_one = helper.make_tensor(
        name="shape_one",
        data_type=TensorProto.INT64,
        dims=[1],
        vals=[1]
    )

    # Index tensor for gather: scalar index 0
    gather_index = helper.make_tensor(
        name="gather_index",
        data_type=TensorProto.INT64,
        dims=[],  # scalar index
        vals=[0]
    )

    # First Reshape: [1,1] tensor -> scalar
    reshape1 = helper.make_node(
        "Reshape",
        inputs=["input", "shape_to_scalar"],
        outputs=["scalar_val"],
        name="reshape_to_scalar"
    )

    # Second Reshape: scalar -> [1] (with optimization, remains scalar)
    reshape2 = helper.make_node(
        "Reshape",
        inputs=["scalar_val", "shape_one"],
        outputs=["reshaped_scalar"],
        name="reshape_scalar_to_1"
    )

    # Gather: gather from scalar at index 0 (should pass through the scalar)
    gather = helper.make_node(
        "Gather",
        inputs=["reshaped_scalar", "gather_index"],
        outputs=["output"],
        axis=0,
        name="gather_from_scalar"
    )

    # Create the graph
    graph_def = helper.make_graph(
        [reshape1, reshape2, gather],
        "gather_scalar_input_model",
        [input_tensor],
        [output_tensor],
        [shape_to_scalar, shape_one, gather_index]
    )

    # Create the model
    model_def = helper.make_model(
        graph_def,
        producer_name="gather_scalar_input",
        opset_imports=[helper.make_operatorsetid("", 16)]
    )

    # Save the model
    onnx.save(model_def, "gather_scalar_input.onnx")
    print("Model exported successfully to gather_scalar_input.onnx")
    print("Model structure: Reshape([1,1] -> scalar) -> Reshape(scalar, [1]) -> Gather(scalar, 0) -> scalar")

    # Verify with ONNX Runtime or reference evaluator
    try:
        from onnx.reference import ReferenceEvaluator

        test_input = np.array([[123.456]], dtype=np.float32)
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
