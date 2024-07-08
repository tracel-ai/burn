from typing import Any
import numpy
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

    onnx.save(onnx_model, f"{name}.onnx")


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
    input_tensor = numpy.arange(1, 7).reshape(3, 2)
    pads = numpy.array([1, 2, 3, 4], dtype="int")
    constant_value = 0
    feeds = {
        "input_tensor": input_tensor,
        "pads": pads,
        "constant_value": constant_value,
    }
    expected = numpy.array(
        [
            [
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 2, 0, 0, 0, 0],
                [0, 0, 3, 4, 0, 0, 0, 0],
                [0, 0, 5, 6, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
            ]
        ]
    )

    TestCase("test_positive_constant_pads", feeds, expected).test_model(model)


def test_1d_input(model: ModelProto) -> None:
    input_tensor = numpy.arange(1, 5)
    pads = numpy.array([1, 2], dtype="int")
    constant_value = 0
    feeds = {
        "input_tensor": input_tensor,
        "pads": pads,
        "constant_value": constant_value,
    }
    expected = numpy.array([[0, 1, 2, 3, 4, 0, 0]])

    TestCase("test_1d_input", feeds, expected).test_model(model)


def run_tests(model: ModelProto) -> None:
    test_positive_pads(model)
    test_1d_input(model)
    # TODO: test_negative_pads
    # TODO: support other modes: reflect, edge, wrap
