#!/usr/bin/env python3

# used to generate model: onnx-tests/tests/resize/resize.onnx

import onnx
from onnx import helper, TensorProto

def main() -> None:
    # Define the input tensor
    input_tensor = helper.make_tensor_value_info("input_tensor", TensorProto.FLOAT, [1, 1, 4, 4])

    # Sizes
    sizes_val = [1, 1, 2, 2]  # Downsample to [1, 1, 2, 2]
    sizes_tensor = helper.make_tensor(
        name="sizes",
        data_type=TensorProto.INT64,
        dims=[len(sizes_val)],
        vals=sizes_val,
    )
    sizes_node = helper.make_node(
        "Constant",
        name="sizes_constant",
        inputs=[],
        outputs=["sizes"],
        value=sizes_tensor,
    )

    # Define the Resize node that uses the outputs from the constant nodes
    resize_node = helper.make_node(
        "Resize",
        name="resize_node",
        inputs=["input_tensor", "", "", "sizes"],
        outputs=["output"],
        mode="linear",
        coordinate_transformation_mode="half_pixel",  # Example mode
        antialias=1,  # Antialiasing enabled
    )

    # Create the graph
    graph_def = helper.make_graph(
        nodes=[sizes_node, resize_node],
        name="ResizeGraph",
        inputs=[input_tensor],
        outputs=[
            helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1, 2, 2])
        ],
    )

    # Create the model
    model_def = helper.make_model(graph_def, producer_name="resize")

    # Save the model to a file
    onnx.save(model_def, "resize.onnx")


if __name__ == "__main__":
    main()
