#!/usr/bin/env python3

# used to generate model: flatten_2d.onnx

# Adapted from https://github.com/onnx/onnx/blob/main/docs/Operators.md#flatten

import onnx
import onnx.helper

def build_model():
    return onnx.helper.make_model(
        opset_imports=[onnx.helper.make_operatorsetid("", 16)],
        graph=onnx.helper.make_graph(name="FlattenGraph", nodes=[
           onnx.helper.make_node(
                "Flatten",
                inputs=["a"],
                outputs=["b"],
                axis=2,
            ),
        ],
        inputs=[
            onnx.helper.make_value_info(
                name="a",
                type_proto=onnx.helper.make_tensor_type_proto(
                    elem_type=onnx.TensorProto.FLOAT, shape=[2, 3, 4, 5]
                ),
            ),
        ],
        outputs=[
            onnx.helper.make_value_info(
                name="b",
                type_proto=onnx.helper.make_tensor_type_proto(
                    elem_type=onnx.TensorProto.FLOAT, shape=[2 * 3, 4 * 5]
                ),
            )
        ]),
    )

if __name__ == "__main__":
    onnx_model = build_model()
    file_name = "flatten_2d.onnx"

    # Ensure valid ONNX:
    onnx.checker.check_model(onnx_model)

    onnx.save(onnx_model, file_name)
