#!/usr/bin/env python3

# used to generate model: sinh.onnx

# torch doesn't support exporting sinh operation to ONNX
# Hence this model is exported using onnx directly

import onnx


def build_model():
    return onnx.helper.make_model(
        ir_version=8,
        opset_imports=[onnx.helper.make_operatorsetid("", 16)],
        graph=onnx.helper.make_graph(
            name="main_graph",
            nodes=[
                onnx.helper.make_node(
                    "Sinh",
                    inputs=["input1"],
                    outputs=["output1"],
                    name="/Sinh"
                ),
            ],
            inputs=[
                onnx.helper.make_value_info(
                    name="input1",
                    type_proto=onnx.helper.make_tensor_type_proto(
                        elem_type=onnx.TensorProto.FLOAT, shape=[1, 1, 1, 4]
                    ),
                ),
            ],
            outputs=[
                onnx.helper.make_value_info(
                    name="output1",
                    type_proto=onnx.helper.make_tensor_type_proto(
                        elem_type=onnx.TensorProto.FLOAT, shape=[1, 1, 1, 4]
                    ),
                )
            ]
        ),
    )


def main():
    onnx_model = build_model()
    file_name = "sinh.onnx"

    # Ensure valid ONNX:
    onnx.checker.check_model(onnx_model)

    onnx.save(onnx_model, file_name)
    print(f"Finished exporting model to {file_name}") 

if __name__ == "__main__":
    main()

