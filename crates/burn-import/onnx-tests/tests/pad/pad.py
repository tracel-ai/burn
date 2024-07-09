#!/usr/bin/env python3

# used to generate model: onnx-tests/tests/pad/pad.onnx

import numpy
from onnx import TensorProto, numpy_helper
from onnx.helper import make_tensor_value_info

from pad_lib import build_test_save


def get_initializers() -> list[TensorProto]:
    constant_value = numpy_helper.from_array(
        numpy.array([0.0]).astype(numpy.float32), name="constant_value"
    )

    return [constant_value]


def main() -> None:
    name = "pad"

    inputs = [
        make_tensor_value_info("input_tensor", TensorProto.FLOAT, [None, None]),
        make_tensor_value_info("pads", TensorProto.FLOAT, [None, None]),
    ]
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
