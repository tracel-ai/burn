#!/usr/bin/env python3

# used to generate model: is_inf_pos_only.onnx

import numpy as np
import onnx
import onnx.helper
from onnx import TensorProto
from onnx.reference import ReferenceEvaluator


def build_model():
    # Define the graph inputs and outputs
    input = onnx.helper.make_tensor_value_info(
        'input', TensorProto.FLOAT, [1, 4])
    output = onnx.helper.make_tensor_value_info(
        'output', TensorProto.BOOL, [1, 4])

    # Create the GroupNormalization node
    is_inf = onnx.helper.make_node(
        "IsInf",
        inputs=["input"],
        outputs=["output"],
        name="IsInfNode",
        detect_negative=0,
    )

    # Create the graph
    graph = onnx.helper.make_graph(
        [is_inf],
        'IsInfModel',
        [input],
        [output],
    )

    # Create the model
    model = onnx.helper.make_model(
        opset_imports=[onnx.helper.make_operatorsetid("", 21)],
        graph=graph,
        producer_name='ONNX_Generator',
    )

    return model


if __name__ == "__main__":
    # Set seed and precision
    np.random.seed(42)
    np.set_printoptions(precision=8)

    # Build model
    test_input = np.random.randn(1, 4).round(2)
    onnx_model = build_model()
    file_name = "is_inf_pos_only.onnx"

    # Ensure valid ONNX and save
    onnx.checker.check_model(onnx_model)
    onnx.save(onnx_model, file_name)
    print(f"Finished exporting model to {file_name}")

    # Output some test data for use in the test
    print(f"Test input data: {repr(test_input)}")
    print(f"Test input data shape: {test_input.shape}")
    session = ReferenceEvaluator("is_inf_pos_only.onnx", verbose=1)
    test_output, = session.run(None, {"input": test_input})
    print(f"Test output: {repr(test_output)}")
    print(f"Test output shape: {test_output.shape}")
