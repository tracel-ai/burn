#!/usr/bin/env python3

# used to generate model: random_normal.onnx

# torch doesn't generate RandomNormal operations in ONNX,
# but always uses RandomNormalLike.
# Hence this model is exported using onnx directly

import onnx
import onnx.helper


def build_model():
    return onnx.helper.make_model(
        ir_version=8,
        opset_imports=[onnx.helper.make_operatorsetid("", 16)],
        graph=onnx.helper.make_graph(name="main_graph", nodes=[
            onnx.helper.make_node(
                "RandomNormal",
                inputs=[],
                outputs=["output1"],
                name="/RandomNormal",
                mean=2.0,
                scale=1.5,
                shape=[2, 3]
            ),
        ],
        inputs=[],
        outputs=[
            onnx.helper.make_value_info(
                name="output1",
                type_proto=onnx.helper.make_tensor_type_proto(
                    elem_type=onnx.TensorProto.FLOAT, shape=[2, 3]
                ),
            )
        ]),
    )


def main():
    onnx_model = build_model()
    file_name = "random_normal.onnx"

    onnx.save(onnx_model, file_name)


if __name__ == "__main__":
    main()
