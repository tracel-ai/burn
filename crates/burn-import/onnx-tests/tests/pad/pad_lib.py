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
) -> None:
    onnx_model = build_model(
        name=name,
        inputs=inputs,
        outputs=outputs,
        initializers=initializers,
    )

    run_tests(onnx_model)

    onnx.save(onnx_model, f"{name}.onnx")


def build_model(
    name: str,
    inputs: list[ValueInfoProto],
    outputs: list[ValueInfoProto],
    initializers: list[TensorProto] = [],
) -> ModelProto:
    node_inputs = [input.name for input in inputs + initializers]
    node_outputs = [output.name for output in outputs]

    node = make_node(
        name.capitalize(),
        inputs=node_inputs,
        outputs=node_outputs,
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

    return onnx_model


class TestCase:
    def __init__(self, feeds: dict[str, numpy.ndarray], expected: numpy.ndarray):
        self.feeds = feeds
        self.expected = expected

    def test_model(self, model: ModelProto):
        sess = ReferenceEvaluator(model)

        result = sess.run(None, self.feeds)

        if not numpy.array_equal(numpy.array(result), self.expected):
            raise Exception("Result not as expected!")


def test_positive_constant_pads(model: ModelProto) -> None:
    feeds = {
        "input_tensor": numpy.array(
            [
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                [11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0],
                [21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0],
                [31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0],
                [41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0, 49.0, 50.0],
            ]
        ),
        "starts": numpy.array([-5, 0], dtype="int"),
        "ends": numpy.array([3, -5], dtype="int"),
        "axes": numpy.array([0, 1], dtype="int"),
        "steps": numpy.array([1, 1], dtype="int"),
    }
    expected = numpy.array(
        [
            [
                [1.0, 2.0, 3.0, 4.0, 5.0],
                [11.0, 12.0, 13.0, 14.0, 15.0],
                [21.0, 22.0, 23.0, 24.0, 25.0],
            ]
        ]
    )

    TestCase(feeds, expected).test_model(model)


def run_tests(model: ModelProto) -> None:
    test_positive_constant_pads(model)
