#!/usr/bin/env python3

# used to generate model: onnx-tests/tests/slice/slice_with_steps.onnx

import onnx
from onnx import helper, TensorProto


def main() -> None:
    # Starts
    starts_val = [0, 1, 0]
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

    # Ends
    ends_val = [10, 8, 12]
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

    # Axes
    axes_val = [0, 1, 2]
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

    # Steps - non-trivial values
    steps_val = [2, 3, -1]  # step=2 for dim0, step=3 for dim1, step=-1 (reverse) for dim2
    steps_tensor = helper.make_tensor(
        name="steps",
        data_type=TensorProto.INT64,
        dims=[len(steps_val)],
        vals=steps_val,
    )
    steps_node = helper.make_node(
        "Constant",
        name="steps_constant",
        inputs=[],
        outputs=["steps"],
        value=steps_tensor,
    )

    # Define the Slice node that uses the outputs from the constant nodes
    slice_node = helper.make_node(
        "Slice",
        name="slice_node",
        inputs=["input_tensor", "starts", "ends", "axes", "steps"],
        outputs=["output"],
    )

    # Create the graph
    graph_def = helper.make_graph(
        nodes=[starts_node, ends_node, axes_node, steps_node, slice_node],
        name="SliceWithStepsGraph",
        inputs=[
            helper.make_tensor_value_info("input_tensor", TensorProto.FLOAT, [10, 10, 12]),
        ],
        outputs=[
            helper.make_tensor_value_info("output", TensorProto.FLOAT, [5, 3, 12])
            # dim0: (10-0+1)/2 = 5 elements (indices: 0,2,4,6,8)
            # dim1: (8-1+2)/3 = 3 elements (indices: 1,4,7)
            # dim2: 12 elements with step=-1 (reverse order)
        ],
    )

    # Create the model
    model_def = helper.make_model(graph_def, producer_name="slice_with_steps")

    # Save the model to a file
    onnx.save(model_def, "slice_with_steps.onnx")


if __name__ == "__main__":
    main()