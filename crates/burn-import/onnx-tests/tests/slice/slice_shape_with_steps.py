#!/usr/bin/env python3

# used to generate model: onnx-tests/tests/slice/slice_shape_with_steps.onnx

import onnx
from onnx import helper, TensorProto


def main() -> None:
    # Create a Shape node to extract the shape
    shape_node = helper.make_node(
        "Shape",
        inputs=["input_tensor"],
        outputs=["shape_output"],
    )

    # Create constant nodes for slice parameters
    # Start at index 0
    starts_tensor = helper.make_tensor(
        name="starts",
        data_type=TensorProto.INT64,
        dims=[1],
        vals=[0],
    )
    starts_node = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["starts"],
        value=starts_tensor,
    )

    # End at index 6
    ends_tensor = helper.make_tensor(
        name="ends",
        data_type=TensorProto.INT64,
        dims=[1],
        vals=[6],
    )
    ends_node = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["ends"],
        value=ends_tensor,
    )

    # Axes is 0 (we're slicing the first dimension of the shape tensor)
    axes_tensor = helper.make_tensor(
        name="axes",
        data_type=TensorProto.INT64,
        dims=[1],
        vals=[0],
    )
    axes_node = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["axes"],
        value=axes_tensor,
    )

    # Steps is 2 (take every other element)
    steps_tensor = helper.make_tensor(
        name="steps",
        data_type=TensorProto.INT64,
        dims=[1],
        vals=[2],
    )
    steps_node = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["steps"],
        value=steps_tensor,
    )

    # Create the Slice node that operates on the shape output
    slice_node = helper.make_node(
        "Slice",
        inputs=["shape_output", "starts", "ends", "axes", "steps"],
        outputs=["output"],
    )

    # Create the graph
    graph_def = helper.make_graph(
        nodes=[shape_node, starts_node, ends_node, axes_node, steps_node, slice_node],
        name="SliceShapeWithStepsGraph",
        inputs=[
            helper.make_tensor_value_info("input_tensor", TensorProto.FLOAT, [2, 3, 4, 5, 6, 7]),
        ],
        outputs=[
            helper.make_tensor_value_info("output", TensorProto.INT64, [3])  # Will output [2, 4, 6]
        ],
    )

    # Create the model
    model_def = helper.make_model(graph_def, producer_name="slice_shape_with_steps")

    # Save the model to a file
    onnx.save(model_def, "slice_shape_with_steps.onnx")
    print("Successfully created slice_shape_with_steps.onnx")


if __name__ == "__main__":
    main()