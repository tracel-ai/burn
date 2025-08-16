#!/usr/bin/env python3

# used to generate model: onnx-tests/tests/reduce/reduce_sum_square.onnx

import numpy as np
import onnx
import onnx.helper
from onnx import TensorProto
from onnx.reference import ReferenceEvaluator


def build_model():
    # Define the graph inputs and outputs
    input = onnx.helper.make_tensor_value_info(
        'input', TensorProto.FLOAT, [1, 1, 2, 4])
    output1 = onnx.helper.make_tensor_value_info(
        'output1', TensorProto.FLOAT, [1])
    output2 = onnx.helper.make_tensor_value_info(
        'output2', TensorProto.FLOAT, [1, 1, 2, 4])
    output3 = onnx.helper.make_tensor_value_info(
        'output3', TensorProto.FLOAT, [1, 1, 2, 1])
    output4 = onnx.helper.make_tensor_value_info(
        'output4', TensorProto.FLOAT, [1, 2, 4])
    output5 = onnx.helper.make_tensor_value_info(
        'output5', TensorProto.FLOAT, [1, 2, 4])

    # ReduceSumSquare, keepdims=0, axes=None
    reduce_sum_square1 = onnx.helper.make_node(
        "ReduceSumSquare",
        inputs=["input"],
        outputs=["output1"],
        name="ReduceSumSquare1",
        keepdims=0,
        axes=None
    )
    # ReduceSumSquare, keepdims=1, axes=[1]
    reduce_sum_square2 = onnx.helper.make_node(
        "ReduceSumSquare",
        inputs=["input"],
        outputs=["output2"],
        name="ReduceSumSquare2",
        keepdims=1,
        axes=[1]
    )
    # ReduceSumSquare, keepdims=1, axes=[-1]
    reduce_sum_square3 = onnx.helper.make_node(
        "ReduceSumSquare",
        inputs=["input"],
        outputs=["output3"],
        name="ReduceSumSquare3",
        keepdims=1,
        axes=[-1]
    )
    # ReduceSumSquare, keepdims=0, axes=[0]
    reduce_sum_square4 = onnx.helper.make_node(
        "ReduceSumSquare",
        inputs=["input"],
        outputs=["output4"],
        name="ReduceSumSquare4",
        keepdims=0,
        axes=[0]
    )
    # ReduceSumSquare, keepdims=0, axes=[0, 2]
    reduce_sum_square5 = onnx.helper.make_node(
        "ReduceSumSquare",
        inputs=["input"],
        outputs=["output5"],
        name="ReduceSumSquare5",
        keepdims=0,
        axes=[0, 2]
    )

    # Create the graph
    graph = onnx.helper.make_graph(
        [reduce_sum_square1, reduce_sum_square2,
         reduce_sum_square3, reduce_sum_square4, reduce_sum_square5],
        'ReduceSumSquareModel',
        [input],
        [output1, output2, output3, output4, output5],
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
    test_input = np.array([[[
        [1.0, 4.0, 9.0, 25.0],
        [2.0, 5.0, 10.0, 26.0],
    ]]])
    onnx_model = build_model()
    file_name = "reduce_sum_square.onnx"

    # Ensure valid ONNX and save
    onnx.checker.check_model(onnx_model)
    onnx.save(onnx_model, file_name)
    print(f"Finished exporting model to {file_name}")

    # Output some test data for use in the test
    print(f"Test input data:\n{repr(test_input)}")
    print(f"Test input data shape: {test_input.shape}")
    session = ReferenceEvaluator(file_name, verbose=1)
    outputs = session.run(None, {"input": test_input})
    print("Test output data:", *outputs, sep="\n")
