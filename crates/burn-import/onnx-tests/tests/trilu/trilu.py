#!/usr/bin/env python3

import onnx
import onnx.helper
import onnx.checker

def build_model():
    # Define the input tensor as a graph input
    input_tensor = onnx.helper.make_tensor_value_info(
        name="input_tensor",
        elem_type=onnx.TensorProto.FLOAT,
        shape=[2, 2]  # Change this as needed for your use case
    )

    output_tensor = onnx.helper.make_tensor_value_info(
        name="output_tensor",
        elem_type=onnx.TensorProto.FLOAT,
        shape=[2, 2]  # Change this as needed for your output shape
    )

    # Create the Trilu node
    # The attributes specify upper (True) and the diagonal k (0)
    trilu_node = onnx.helper.make_node(
        "Trilu",  # or "Triu" for upper triangular
        inputs=["input_tensor"],
        outputs=["output_tensor"],
        name="trilu_node",
        **{
            "upper": True,  # Set to False for lower triangular
            "k": 0  # Modify k based on your requirements
        }
    )

    # Build the graph
    graph = onnx.helper.make_graph(
        nodes=[trilu_node],
        name="main_graph",
        inputs=[input_tensor],
        outputs=[output_tensor]
    )

    # Build the model
    model = onnx.helper.make_model(
        graph,
        ir_version=8,
        opset_imports=[onnx.helper.make_operatorsetid("", 16)]  # Ensure the opset version supports Trilu
    )

    return model

def main():
    onnx_model = build_model()

    # Perform shape inference
    onnx_model = onnx.shape_inference.infer_shapes(onnx_model)

    file_name = "trilu.onnx"
    onnx.save(onnx_model, file_name)
    # onnx.checker.check_model(onnx_model)
    print(f"ONNX model saved as {file_name}")

if __name__ == "__main__":
    main()