#!/usr/bin/env python3

import onnx
import onnx.helper
import onnx.checker


def build_model():
    # Define the input tensor as a graph input
    input_tensor = onnx.helper.make_tensor_value_info(
        name="input_tensor",
        elem_type=onnx.TensorProto.FLOAT,
        shape=[2, 2]
    )

    output_tensor = onnx.helper.make_tensor_value_info(
        name="output_tensor",
        elem_type=onnx.TensorProto.FLOAT,
        shape=[4, 4]
    )

    # Define the shape tensor for tiling as an initializer
    shape_tensor = onnx.helper.make_tensor(
        name="shape_tensor",
        data_type=onnx.TensorProto.INT64,
        dims=[2],
        vals=[2, 2]
    )
    # Create the Tile node
    tile_node = onnx.helper.make_node(
        "Tile",
        inputs=["input_tensor", "shape_tensor"],
        outputs=["output_tensor"]
    )

    # Build the graph
    graph = onnx.helper.make_graph(
        nodes=[tile_node],
        name="main_graph",
        inputs=[input_tensor],
        outputs=[output_tensor],
        initializer=[shape_tensor]
    )

    # Build the model
    model = onnx.helper.make_model(
        graph,
        ir_version=8,
        opset_imports=[onnx.helper.make_operatorsetid("", 16)]
    )

    return model


def main():
    onnx_model = build_model()

    onnx_model = onnx.shape_inference.infer_shapes(onnx_model)

    file_name = "tile.onnx"
    onnx.save(onnx_model, file_name)
    onnx.checker.check_model(onnx_model)
    print(f"ONNX model saved as {file_name}")


if __name__ == "__main__":
    main()
