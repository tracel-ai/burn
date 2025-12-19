#!/usr/bin/env python3

# used to generate model: gather_negative_idx.onnx

import onnx
from onnx import numpy_helper
import numpy as np


def build_model():
    neg_idx = numpy_helper.from_array(np.array(-1, dtype=np.int64), name="neg_idx")

    return onnx.helper.make_model(
        ir_version=8,
        opset_imports=[onnx.helper.make_operatorsetid("", 16)],
        graph=onnx.helper.make_graph(
            name="main_graph",
            nodes=[
                onnx.helper.make_node(
                    "Gather",
                    inputs=["input", "neg_idx"],
                    outputs=["output"],
                    name="/Gather",
                    axis=0,
                ),
            ],
            inputs=[
                onnx.helper.make_value_info(
                    name="input",
                    type_proto=onnx.helper.make_tensor_type_proto(
                        elem_type=onnx.TensorProto.FLOAT, shape=[4, 3]
                    ),
                ),
            ],
            outputs=[
                onnx.helper.make_value_info(
                    name="output",
                    type_proto=onnx.helper.make_tensor_type_proto(
                        elem_type=onnx.TensorProto.FLOAT, shape=[3]
                    ),
                )
            ],
            initializer=[neg_idx],
        ),
    )


def main():
    onnx_model = build_model()
    file_name = "gather_negative_idx.onnx"

    onnx.checker.check_model(onnx_model)
    onnx.save(onnx_model, file_name)

    print(f"Finished exporting model to {file_name}")

    print("\n// Test data:")
    print("// Input shape: [4, 3]")
    print("// Index: -1 (constant, should gather last row)")
    print("// Expected output: [10, 11, 12]")


if __name__ == "__main__":
    main()