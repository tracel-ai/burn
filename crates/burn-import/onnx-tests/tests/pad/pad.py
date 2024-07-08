#!/usr/bin/env python3

# used to generate model: onnx-tests/tests/pad/pad.onnx

import numpy
from onnx import TensorProto, numpy_helper
from onnx.helper import make_tensor_value_info

from pad_lib import build_test_save


def get_initializers() -> list[TensorProto]:
    pads = numpy_helper.from_array(
        numpy.array([1, 2, 3, 4]).astype(numpy.int64), name="pads"
    )
    constant_value = numpy_helper.from_array(
        numpy.array([0]).astype(numpy.int64), name="constant_value"
    )

    return [pads, constant_value]


def main() -> None:
    name = "pad"

    inputs = [make_tensor_value_info("input_tensor", TensorProto.INT64, [None, None])]
    outputs = [make_tensor_value_info("output", TensorProto.INT64, [None, None])]
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
