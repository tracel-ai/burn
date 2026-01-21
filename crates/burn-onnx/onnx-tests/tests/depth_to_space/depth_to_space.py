#!/usr/bin/env python3

# used to generate models: depth_to_space_*.onnx

import numpy as np
import onnx
import onnx.helper
from onnx import TensorProto
from onnx.reference import ReferenceEvaluator


def build_model(mode):
    # Define the graph inputs and outputs
    input = onnx.helper.make_tensor_value_info(
        'input', TensorProto.FLOAT, [2, 4, 2, 3])
    output = onnx.helper.make_tensor_value_info(
        'output', TensorProto.FLOAT, [2, 1, 4, 6])

    # Create the DepthToSpace node
    depth_to_space = onnx.helper.make_node(
        "DepthToSpace",
        inputs=["input"],
        outputs=["output"],
        name="DepthToSpaceNode",
        mode=mode,
        blocksize=2,
    )

    # Create the graph
    graph = onnx.helper.make_graph(
        [depth_to_space],
        f'DepthToSpace{mode}Model',
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


def export_onnx_model(mode):
    print(f"Building model with mode: {mode}...")

    # Build model
    test_input = np.random.randn(2, 4, 2, 3).round(2)
    onnx_model = build_model(mode)
    file_name = f"depth_to_space_{mode.lower()}.onnx"

    # Ensure valid ONNX and save
    onnx.checker.check_model(onnx_model)
    onnx.save(onnx_model, file_name)
    print(f"Finished exporting model to {file_name}")

    # Output some test data for use in the test
    print(f"Test input data:\n{repr(test_input)}")
    print(f"Test input data shape: {test_input.shape}")
    session = ReferenceEvaluator(file_name, verbose=1)
    test_output, = session.run(None, {"input": test_input})
    print(f"Test output:\n{repr(test_output)}")
    print(f"Test output shape: {test_output.shape}")
    print("\n\n")


if __name__ == "__main__":
    # Set seed and precision
    np.random.seed(42)
    np.set_printoptions(precision=8)

    # Export models for DCR and CRD modes
    export_onnx_model('DCR')
    export_onnx_model('CRD')
