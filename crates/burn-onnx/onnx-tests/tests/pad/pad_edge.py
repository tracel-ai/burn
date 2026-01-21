#!/usr/bin/env python3

# used to generate model: onnx-tests/tests/pad/pad_edge.onnx

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
    # Create a simple edge padding model
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
        mode="edge",
    )

    graph = make_graph(
        nodes=[node],
        name="PadEdgeGraph",
        inputs=inputs,
        outputs=outputs,
        initializer=initializers,
    )

    onnx_model = make_model(graph)
    check_model(onnx_model)

    # Test the model
    test_edge_padding(onnx_model)

    onnx.save(onnx_model, Path(__file__).with_name("pad_edge.onnx"))
    print("Generated pad_edge.onnx")


def test_edge_padding(model) -> None:
    """Test edge padding with a simple 2x3 input."""
    sess = ReferenceEvaluator(model)

    # Input: 2x3 tensor
    # [[1, 2, 3],
    #  [4, 5, 6]]
    input_tensor = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0]
    ], dtype=np.float32)

    result = sess.run(None, {"input_tensor": input_tensor})[0]

    # Expected with edge padding (1,1,1,1):
    # Edge replicates the boundary values
    expected = np.array([
        [1.0, 1.0, 2.0, 3.0, 3.0],
        [1.0, 1.0, 2.0, 3.0, 3.0],
        [4.0, 4.0, 5.0, 6.0, 6.0],
        [4.0, 4.0, 5.0, 6.0, 6.0],
    ], dtype=np.float32)

    if not np.allclose(result, expected):
        print(f"Expected:\n{expected}")
        print(f"Got:\n{result}")
        raise Exception("Edge padding test failed")

    print("Edge padding test passed!")


if __name__ == "__main__":
    main()
