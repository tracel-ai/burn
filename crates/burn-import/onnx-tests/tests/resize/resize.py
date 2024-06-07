#!/usr/bin/env python3

# used to generate model: onnx-tests/tests/resize/resize.onnx

import onnx
from onnx import helper, TensorProto

def main() -> None:
    input_tensor = helper.make_tensor_value_info("input_tensor", TensorProto.FLOAT, [1, 1, 4, 4])
    sizes_tensor = helper.make_tensor_value_info("sizes", TensorProto.INT64, [4])

    resize_node = helper.make_node(
        "Resize",
        name="resize_node",
        inputs=["input_tensor", "", "", "sizes"],
        outputs=["output"],
        mode="linear",
    )

    graph_def = helper.make_graph(
        nodes=[resize_node],
        name="ResizeGraph",
        inputs=[input_tensor, sizes_tensor],
        outputs=[
            helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1, 2, 2])
        ],
    )

    model_def = helper.make_model(graph_def, producer_name="resize")

    onnx.save(model_def, "resize.onnx")


if __name__ == "__main__":
    main()
