#!/usr/bin/env python3

import onnx
import onnx.helper
import onnx.checker


def build_model():
    # Define the input tensor as a graph input
    data = onnx.helper.make_tensor_value_info(
        name="data",
        elem_type=onnx.TensorProto.FLOAT,
        shape=[3, 5]  # Example shape
    )

    indices = onnx.helper.make_tensor_value_info(
        name="indices",
        elem_type=onnx.TensorProto.INT64,
        shape=[3, 2]  # Example shape
    )

    updates = onnx.helper.make_tensor_value_info(
        name="updates",
        elem_type=onnx.TensorProto.FLOAT,
        shape=[3, 2]  # Example shape
    )

    output = onnx.helper.make_tensor_value_info(
        name="output",
        elem_type=onnx.TensorProto.FLOAT,
        shape=[3, 5]  # Same shape as data
    )

    # Create the Scatter node
    scatter_node = onnx.helper.make_node(
        "Scatter",
        inputs=["data", "indices", "updates"],
        outputs=["output"],
        axis=1  # Example axis
    )

    # Build the graph
    graph = onnx.helper.make_graph(
        nodes=[scatter_node],
        name="scatter_graph",
        inputs=[data, indices, updates],
        outputs=[output]
    )

    # Build the model
    model = onnx.helper.make_model(
        graph,
        ir_version=8,
        opset_imports=[onnx.helper.make_operatorsetid("", 10)]
    )

    return model


def main():
    onnx_model = build_model()

    onnx_model = onnx.shape_inference.infer_shapes(onnx_model)

    file_name = "scatter.onnx"
    onnx.save(onnx_model, file_name)
    onnx.checker.check_model(onnx_model)
    print(f"ONNX model saved as {file_name}")


if __name__ == "__main__":
    main()
