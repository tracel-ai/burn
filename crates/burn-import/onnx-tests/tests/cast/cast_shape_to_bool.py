#!/usr/bin/env python3

import numpy as np
import onnx
from onnx import helper, TensorProto


def main():
    # Create input tensor (3D) for Shape operation
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [2, 3, 4])

    # Create output tensor (1D bool) after Cast
    output_tensor = helper.make_tensor_value_info("output", TensorProto.BOOL, [3])

    # Create Shape node to extract the shape of the input
    shape_node = helper.make_node(
        "Shape",
        inputs=["input"],
        outputs=["shape_output"],
        name="Shape"
    )

    # Create Cast node to cast shape to bool
    cast_node = helper.make_node(
        "Cast",
        inputs=["shape_output"],
        outputs=["output"],
        to=TensorProto.BOOL,  # Cast to bool type
        name="Cast_to_bool"
    )

    # Create the graph
    graph_def = helper.make_graph(
        [shape_node, cast_node],
        "cast_shape_to_bool",
        [input_tensor],
        [output_tensor],
    )

    # Create the model
    model_def = helper.make_model(graph_def, producer_name="cast_shape_to_bool")
    model_def.opset_import[0].version = 17

    # Save the model
    onnx.save(model_def, "cast_shape_to_bool.onnx")
    print("Model exported successfully to cast_shape_to_bool.onnx")
    print("Model structure: Shape -> Cast(to=bool)")
    print("Input shape: [2, 3, 4]")
    print("Output: 1D bool tensor with 3 elements")


if __name__ == "__main__":
    main()