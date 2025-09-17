#!/usr/bin/env python3

import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

def main():
    # Create a squeeze node that takes axes as input (ONNX opset 13+ style)
    # This simulates the FaceNet512 model case

    # Input tensor shape [1, 512, 1, 1]
    input_tensor = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, 512, 1, 1]
    )

    # Output tensor shape [1, 512] after squeezing dims 2 and 3
    output_tensor = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, 512]
    )

    # Create axes as a constant tensor input (ONNX opset 13+ style)
    axes_tensor = numpy_helper.from_array(
        np.array([2, 3], dtype=np.int64),
        name="axes"
    )

    # Create the squeeze node with axes as input
    squeeze_node = helper.make_node(
        "Squeeze",
        inputs=["input", "axes"],  # axes as second input
        outputs=["output"],
        name="squeeze_with_axes_input"
    )

    # Create the graph
    graph_def = helper.make_graph(
        [squeeze_node],
        "squeeze_opset13_test",
        [input_tensor],
        [output_tensor],
        [axes_tensor]  # axes as initializer
    )

    # Create the model with opset 16
    model_def = helper.make_model(
        graph_def,
        producer_name="squeeze_opset13_test",
        opset_imports=[helper.make_opsetid("", 16)]  # Use ONNX opset 16 (required by burn)
    )

    # Save the model
    onnx.save(model_def, "squeeze_opset13_axes_input.onnx")
    print("Model saved as squeeze_opset13_axes_input.onnx")

if __name__ == "__main__":
    main()