#!/usr/bin/env python3

import onnx
import onnx.helper
import onnx.checker
import numpy as np


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

    # Add values for indices and updates tensors
    indices_values = onnx.helper.make_tensor(
        name="indices",
        data_type=onnx.TensorProto.INT64,
        dims=[3, 2],
        vals=np.array([[0, 1], [1, 0], [2, 1]]).astype(
            np.int64).flatten().tolist()
    )

    updates_values = onnx.helper.make_tensor(
        name="updates",
        data_type=onnx.TensorProto.FLOAT,
        dims=[3, 2],
        vals=np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
                      ).astype(np.float32).flatten().tolist()
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
        inputs=[data],
        outputs=[output],
        # Add initializers for indices and updates
        initializer=[indices_values, updates_values]
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
