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
    feeds = {
        "input_tensor": numpy.array(
            [
                [1.0, 1.2],
                [2.3, 3.4],
                [4.5, 5.7],
            ]
        ),
        "pads": numpy.array([0, 2, 0, 0], dtype="int"),
        "constant_value": "-1.0",
    }
    expected = numpy.array(
        [
            [
                [-1.0, -1.0, 1.0, 1.2],
                [-1.0, -1.0, 2.3, 3.4],
                [-1.0, -1.0, 4.5, 5.7],
            ]
        ]
    )

    TestCase("test_positive_constant_pads", feeds, expected).test_model(model)


def run_tests(model: ModelProto) -> None:
    test_positive_pads(model)
    # TODO: test_negative_pads
