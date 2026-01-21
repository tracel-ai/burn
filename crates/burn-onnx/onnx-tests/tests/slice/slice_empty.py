#!/usr/bin/env python3

# used to generate model: onnx-tests/tests/slice/slice_empty.onnx
#
# Tests ONNX Slice with empty range (start == end).
# This produces a tensor with size 0 in the sliced dimension.

import numpy as np
import onnx
from onnx import helper, TensorProto
from onnx.reference import ReferenceEvaluator

# ONNX opset version to use for model generation
OPSET_VERSION = 16


def main() -> None:
    # Starts - slice from index 2
    starts_val = [2]
    starts_tensor = helper.make_tensor(
        name="starts",
        data_type=TensorProto.INT64,
        dims=[len(starts_val)],
        vals=starts_val,
    )
    starts_node = helper.make_node(
        "Constant",
        name="starts_constant",
        inputs=[],
        outputs=["starts"],
        value=starts_tensor,
    )

    # Ends - end at index 2 (same as start = empty slice)
    ends_val = [2]
    ends_tensor = helper.make_tensor(
        name="ends",
        data_type=TensorProto.INT64,
        dims=[len(ends_val)],
        vals=ends_val,
    )
    ends_node = helper.make_node(
        "Constant",
        name="ends_constant",
        inputs=[],
        outputs=["ends"],
        value=ends_tensor,
    )

    # Axes - slice on dimension 0
    axes_val = [0]
    axes_tensor = helper.make_tensor(
        name="axes",
        data_type=TensorProto.INT64,
        dims=[len(axes_val)],
        vals=axes_val,
    )
    axes_node = helper.make_node(
        "Constant",
        name="axes_constant",
        inputs=[],
        outputs=["axes"],
        value=axes_tensor,
    )

    # Define the Slice node
    slice_node = helper.make_node(
        "Slice",
        name="slice_node",
        inputs=["input_tensor", "starts", "ends", "axes"],
        outputs=["output"],
    )

    # Create the graph
    # Input shape: [4, 3], Output shape: [0, 3] (empty first dimension)
    graph_def = helper.make_graph(
        nodes=[starts_node, ends_node, axes_node, slice_node],
        name="SliceEmptyGraph",
        inputs=[
            helper.make_tensor_value_info("input_tensor", TensorProto.FLOAT, [4, 3]),
        ],
        outputs=[
            helper.make_tensor_value_info("output", TensorProto.FLOAT, [0, 3])
        ],
    )

    # Create the model
    model_def = helper.make_model(
        graph_def,
        producer_name="slice_empty",
        opset_imports=[helper.make_operatorsetid("", OPSET_VERSION)]
    )

    # Ensure valid ONNX
    onnx.checker.check_model(model_def)

    # Save the model to a file
    onnx_name = "slice_empty.onnx"
    onnx.save(model_def, onnx_name)
    print(f"Finished exporting model to {onnx_name}")

    # Test the model with sample data
    test_input = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
        [10.0, 11.0, 12.0],
    ], dtype=np.float32)
    print(f"\nTest input shape: {test_input.shape}")
    print(f"Test input:\n{test_input}")

    # Run the model using ReferenceEvaluator
    session = ReferenceEvaluator(onnx_name, verbose=0)
    outputs = session.run(None, {"input_tensor": test_input})

    output = outputs[0]
    print(f"\nTest output shape: {output.shape}")
    print(f"Test output:\n{output}")

    # Verify empty slice: output should have shape [0, 3]
    assert output.shape == (0, 3), f"Expected shape (0, 3), got {output.shape}"
    print("\nEmpty slice verification passed!")


if __name__ == "__main__":
    main()
