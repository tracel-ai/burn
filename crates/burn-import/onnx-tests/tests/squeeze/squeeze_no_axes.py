#!/usr/bin/env python3

# This script generates ONNX model for testing squeeze without axes specified
# When no axes are provided, all dimensions with size 1 should be squeezed

import numpy as np
import onnx
from onnx import helper, TensorProto

def main():
    # Create a tensor with shape [2, 1, 3, 1, 4]
    # When squeezed without axes, should become [2, 3, 4]
    input_tensor = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [2, 1, 3, 1, 4]
    )

    # Output tensor shape [2, 3, 4] after squeezing all dims with size 1
    output_tensor = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [2, 3, 4]
    )

    # Create the squeeze node without axes input
    # This means squeeze all dimensions with size 1
    squeeze_node = helper.make_node(
        "Squeeze",
        inputs=["input"],  # No axes input
        outputs=["output"],
        name="squeeze_all_ones"
    )

    # Create the graph
    graph_def = helper.make_graph(
        [squeeze_node],
        "squeeze_no_axes_test",
        [input_tensor],
        [output_tensor],
    )

    # Create the model with opset 16
    model_def = helper.make_model(
        graph_def,
        producer_name="squeeze_no_axes_test",
        opset_imports=[helper.make_opsetid("", 16)]
    )

    # Save the model
    onnx.save(model_def, "squeeze_no_axes.onnx")
    print("Model saved as squeeze_no_axes.onnx")

if __name__ == "__main__":
    main()