#!/usr/bin/env python3

# used to generate model: gather_shape.onnx

# torch doesn't easily generate Shape into Gather operations in ONNX
# (tensor.size and .shape just return a tuple, no tensor)
# Hence this model is exported using onnx directly

import onnx
import onnx.helper


def build_model():
    return onnx.helper.make_model(
        ir_version=8,
        opset_imports=[onnx.helper.make_operatorsetid("", 16)],
        graph=onnx.helper.make_graph(name="main_graph", nodes=[
            onnx.helper.make_node(
                "Shape",
                inputs=["input1"],
                outputs=["shape1"],
                name="/Shape"
            ),
            onnx.helper.make_node(
                "Gather",
                inputs=["shape1", "input2"],
                outputs=["output1"],
                name="/Gather",
                axis=0
            ),
        ],
        inputs=[
            onnx.helper.make_value_info(
                name="input1",
                type_proto=onnx.helper.make_tensor_type_proto(
                    elem_type=onnx.TensorProto.FLOAT, shape=[2, 3]
                ),
            ),
            onnx.helper.make_value_info(
                name="input2",
                type_proto=onnx.helper.make_tensor_type_proto(
                    elem_type=onnx.TensorProto.INT64, shape=[1]
                ),
            ),

        ],
        outputs=[
            onnx.helper.make_value_info(
                name="output1",
                type_proto=onnx.helper.make_tensor_type_proto(
                    elem_type=onnx.TensorProto.INT64, shape=[1]
                ),
            )
        ]),
    )


def main():
    onnx_model = build_model()
    file_name = "gather_shape.onnx"

    # Ensure valid ONNX:
    onnx.checker.check_model(onnx_model)

    onnx.save(onnx_model, file_name)


if __name__ == "__main__":
    main()
