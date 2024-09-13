#!/usr/bin/env python3

# used to generate model: onnx-tests/tests/pad/pad.onnx

### Helper Functions ###
from pathlib import Path
from typing import Any
import numpy
from numpy.core.multiarray import dtype
import onnx
from onnx import ModelProto, TensorProto, ValueInfoProto
from onnx.reference import ReferenceEvaluator
from onnx.checker import check_model
from onnx.helper import (
    make_model,
    make_node,
    make_graph,
)


def build_test_save(
    name: str,
    inputs: list[ValueInfoProto],
    outputs: list[ValueInfoProto],
    initializers: list[TensorProto] = [],
    attributes: dict[str, Any] = {},
) -> None:
    node_inputs = [input.name for input in inputs + initializers]
    node_outputs = [output.name for output in outputs]

    node = make_node(
        name.capitalize(),
        inputs=node_inputs,
        outputs=node_outputs,
        **attributes,
    )

    graph = make_graph(
        nodes=[node],
        name=f"{name.capitalize()}Graph",
        inputs=inputs,
        outputs=outputs,
        initializer=initializers,
    )

    onnx_model = make_model(graph)
    check_model(onnx_model)

    run_tests(onnx_model)

    onnx.save(onnx_model, Path(__file__).with_name(f"{name}.onnx"))


class TestCase:
    def __init__(
        self, name: str, feeds: dict[str, numpy.ndarray], expected: numpy.ndarray
    ):
        self.name = name
        self.feeds = feeds
        self.expected = expected

    def test_model(self, model: ModelProto):
        sess = ReferenceEvaluator(model)

        result = numpy.array(sess.run(None, self.feeds))

        if not numpy.array_equal(result, self.expected):
            print(
                f"""{self.name}
Expected result: {self.expected}
Got: {result}"""
            )
            raise Exception("Test failed")


def test_positive_pads(model: ModelProto) -> None:
    input_tensor = numpy.arange(1, 7, dtype="float32").reshape(3, 2)
    pads = numpy.array([1, 2, 3, 4], dtype="int")
    constant_value = 0.0
    feeds = {
        "input_tensor": input_tensor,
        "pads": pads,
        "constant_value": constant_value,
    }
    expected = numpy.array(
        [
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 5.0, 6.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        ]
    )

    TestCase("test_positive_constant_pads", feeds, expected).test_model(model)


def test_1d_input(model: ModelProto) -> None:
    input_tensor = numpy.arange(1, 5, dtype="float32")
    pads = numpy.array([1, 2], dtype="int")
    constant_value = 0.0
    feeds = {
        "input_tensor": input_tensor,
        "pads": pads,
        "constant_value": constant_value,
    }
    expected = numpy.array([[0.0, 1.0, 2.0, 3.0, 4.0, 0.0, 0.0]])

    TestCase("test_1d_input", feeds, expected).test_model(model)


def run_tests(model: ModelProto) -> None:
    test_positive_pads(model)
    test_1d_input(model)
    # TODO: test_negative_pads
    # TODO: support other modes: reflect, edge, wrap


### Helper Functions End ###

import numpy
from onnx import TensorProto, numpy_helper
from onnx.helper import make_tensor_value_info


def get_initializers() -> list[TensorProto]:
    pads = numpy_helper.from_array(
        numpy.array([1, 2, 3, 4]).astype(numpy.int64), name="pads"
    )
    constant_value = numpy_helper.from_array(
        numpy.array([0.0]).astype(numpy.float32), name="constant_value"
    )

    return [pads, constant_value]


def main() -> None:
    name = "pad"

    inputs = [make_tensor_value_info("input_tensor", TensorProto.FLOAT, [None, None])]
    outputs = [make_tensor_value_info("output", TensorProto.FLOAT, [None, None])]
    initializers = get_initializers()

    build_test_save(
        name=name,
        inputs=inputs,
        outputs=outputs,
        initializers=initializers,
        attributes={"mode": "constant"},
    )


if __name__ == "__main__":
    main()
