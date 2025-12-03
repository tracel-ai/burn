#!/usr/bin/env python3

# used to generate model: onnx-tests/tests/pad/pad_reflect.onnx

from pathlib import Path
import numpy as np
import onnx
from onnx import TensorProto, numpy_helper
from onnx.reference import ReferenceEvaluator
from onnx.checker import check_model
from onnx.helper import (
    make_model,
    make_node,
    make_graph,
    make_tensor_value_info,
)


def main() -> None:
    # Create a simple reflect padding model
    # Input: 2D tensor, pads as initializer

    # Define inputs
    inputs = [make_tensor_value_info("input_tensor", TensorProto.FLOAT, [None, None])]
    outputs = [make_tensor_value_info("output", TensorProto.FLOAT, [None, None])]

    # Pads: [top, left, bottom, right] for 2D = [1, 1, 1, 1]
    # ONNX format: [begin_dim0, begin_dim1, ..., end_dim0, end_dim1, ...]
    pads = numpy_helper.from_array(
        np.array([1, 1, 1, 1]).astype(np.int64), name="pads"
    )

    initializers = [pads]

    node = make_node(
        "Pad",
        inputs=["input_tensor", "pads"],
        outputs=["output"],
        mode="reflect",
    )

    graph = make_graph(
        nodes=[node],
        name="PadReflectGraph",
        inputs=inputs,
        outputs=outputs,
        initializer=initializers,
    )

    onnx_model = make_model(graph)
    check_model(onnx_model)

    # Test the model
    test_reflect_padding(onnx_model)

    onnx.save(onnx_model, Path(__file__).with_name("pad_reflect.onnx"))
    print("Generated pad_reflect.onnx")


def test_reflect_padding(model) -> None:
    """Test reflect padding with a simple 3x3 input."""
    sess = ReferenceEvaluator(model)

    # Input: 3x3 tensor
    # [[1, 2, 3],
    #  [4, 5, 6],
    #  [7, 8, 9]]
    input_tensor = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0]
    ], dtype=np.float32)

    result = sess.run(None, {"input_tensor": input_tensor})[0]

    # Expected with reflect padding (1,1,1,1):
    # Reflect excludes the edge value
    # Top row: reflect row 1 (index 1) -> [4, 5, 6]
    # Bottom row: reflect row 1 (index 1 from end) -> [4, 5, 6]
    # Left col: reflect col 1 -> [2, 5, 8]
    # Right col: reflect col 1 from end -> [2, 5, 8]
    expected = np.array([
        [5.0, 4.0, 5.0, 6.0, 5.0],
        [2.0, 1.0, 2.0, 3.0, 2.0],
        [5.0, 4.0, 5.0, 6.0, 5.0],
        [8.0, 7.0, 8.0, 9.0, 8.0],
        [5.0, 4.0, 5.0, 6.0, 5.0],
    ], dtype=np.float32)

    if not np.allclose(result, expected):
        print(f"Expected:\n{expected}")
        print(f"Got:\n{result}")
        raise Exception("Reflect padding test failed")

    print("Reflect padding test passed!")


if __name__ == "__main__":
    main()
