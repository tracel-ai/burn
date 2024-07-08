#!/usr/bin/env python3

# used to generate model: onnx-tests/tests/pad/pad.onnx

import numpy
from onnx import TensorProto, numpy_helper
from onnx.helper import make_tensor_value_info

from .pad_lib import save_model


def get_initializers() -> list[TensorProto]:
    constant_value = numpy_helper.from_array(
        numpy.array([0], dtype="int"), name="starts"
    )

    value = numpy.array([-1, -1], dtype="int")
    ends_init = numpy_helper.from_array(value, name="ends")

    value = numpy.arange(2, dtype="int")
    axes_init = numpy_helper.from_array(value, name="axes")

    value = numpy.ones(2, dtype="int")
    steps_init = numpy_helper.from_array(value, name="steps")

    return [constant_value, ends_init, axes_init, steps_init]


def main() -> None:
    name = "pad"
    inputs = [
        make_tensor_value_info("input_tensor", TensorProto.FLOAT, [None, None]),
        make_tensor_value_info("pads", TensorProto.FLOAT, [None, None]),
    ]
    outputs = [make_tensor_value_info("output", TensorProto.FLOAT, [None, None])]
    initializers = get_initializers()

    save_model(
        name=name,
        inputs=inputs,
        outputs=outputs,
        initializers=initializers,
    )


if __name__ == "__main__":
    main()
