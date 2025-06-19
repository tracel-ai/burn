#!/usr/bin/env python3

# used to generate model: size.onnx

import numpy as np
import onnx
import onnx.helper
from onnx import TensorProto
from onnx.reference import ReferenceEvaluator


def build_model():
    # Define the graph inputs and outputs
    input = onnx.helper.make_tensor_value_info(
        'input', TensorProto.FLOAT, [2, 6, 2, 3])
    output = onnx.helper.make_tensor_value_info(
        'output', TensorProto.FLOAT, [1])

    # Create the Size node
    size = onnx.helper.make_node(
        "Size",
        inputs=["input"],
        outputs=["output"],
        name="SizeNode",
    )

    # Create the graph
    graph = onnx.helper.make_graph(
        [size],
        'SizeModel',
        [input],
        [output],
    )

    # Create the model
    model = onnx.helper.make_model(
        opset_imports=[onnx.helper.make_operatorsetid("", 16)],
        graph=graph,
        producer_name='ONNX_Generator',
    )

    return model


if __name__ == "__main__":
    # Set seed and precision
    np.random.seed(42)
    np.set_printoptions(precision=8)

    # Build model
    test_input = np.arange(1*2*3*4*5).reshape(1, 2, 3, 4, 5)
    onnx_model = build_model()
    file_name = "size.onnx"

    # Ensure valid ONNX and save
    onnx.checker.check_model(onnx_model)
    onnx.save(onnx_model, file_name)
    print(f"Finished exporting model to {file_name}")

    # Output some test data for use in the test
    print(f"Test input data shape: {test_input.shape}")
    session = ReferenceEvaluator(file_name, verbose=1)
    test_output, = session.run(None, {"input": test_input})
    print(f"Test output: {repr(test_output)}")
