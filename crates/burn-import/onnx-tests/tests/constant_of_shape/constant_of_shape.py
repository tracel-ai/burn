#!/usr/bin/env python3

# used to generate model: constant_of_shape.onnx

# torch simplifies simple usecases where it can statically determine the shape of the constant
# to use just ONNX constants instead of ConstantOfShape
# Hence this model is exported using onnx directly

import onnx
import onnx.helper


def build_model():
    return onnx.helper.make_model(
        ir_version=8,
        opset_imports=[onnx.helper.make_operatorsetid("", 16)],
        graph=onnx.helper.make_graph(name="main_graph", nodes=[
            onnx.helper.make_node(
                "ConstantOfShape",
                inputs=["input1"],
                outputs=["output1"],
                name="/ConstantOfShape",
                value=onnx.helper.make_tensor("value", data_type=onnx.TensorProto.FLOAT, dims=[1], vals=[1.125])
            ),
        ],
        inputs=[
            onnx.helper.make_value_info(
                name="input1",
                type_proto=onnx.helper.make_tensor_type_proto(
                    elem_type=onnx.TensorProto.INT64, shape=[3]
                ),
            )
        ],
        outputs=[
            onnx.helper.make_value_info(
                name="output1",
                type_proto=onnx.helper.make_tensor_type_proto(
                    elem_type=onnx.TensorProto.FLOAT, shape=[2, 3, 2]
                ),
            )
        ]),
    )


def main():
    onnx_model = build_model()
    file_name = "constant_of_shape.onnx"
    onnx.save(onnx_model, file_name)
    onnx.checker.check_model(file_name)


if __name__ == "__main__":
    main()
