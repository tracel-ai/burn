#!/usr/bin/env python3

# used to generate model: bernoulli.onnx

import onnx
import onnx.helper
from onnx import TensorProto


def build_model():
    # Define the graph inputs and outputs
    input = onnx.helper.make_tensor_value_info(
        'input', TensorProto.FLOAT, [10])
    output = onnx.helper.make_tensor_value_info(
        'output', TensorProto.FLOAT, [10])

    # Create the Bernoulli node
    bernoulli = onnx.helper.make_node(
        "Bernoulli",
        inputs=['input'],
        outputs=["output"],
        name="/Bernoulli",
        dtype=onnx.TensorProto.FLOAT,
    )

    # Create the graph
    graph = onnx.helper.make_graph(
        name="main_graph",
        nodes=[bernoulli],
        inputs=[input],
        outputs=[output],
    )

    # Create the model
    model = onnx.helper.make_model(
        opset_imports=[onnx.helper.make_operatorsetid("", 16)],
        graph=graph,
        producer_name='ONNX_Generator',
    )

    return model


def main():
    # Build model
    onnx_model = build_model()
    file_name = "bernoulli.onnx"

    # Validate & save the model
    onnx.checker.check_model(onnx_model)
    onnx.save(onnx_model, file_name)
    print(f"Finished exporting model to {file_name}")


if __name__ == "__main__":
    main()
