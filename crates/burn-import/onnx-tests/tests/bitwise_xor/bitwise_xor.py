import torch
import onnx


def build_model():
    return onnx.helper.make_model(
        ir_version=8,
        opset_imports=[onnx.helper.make_operatorsetid("", 18)],
        graph=onnx.helper.make_graph(
            name="main_graph",
            nodes=[
                onnx.helper.make_node(
                    "BitwiseXor",
                    inputs=["input1", "input2"],
                    outputs=["output"],
                    name="/BitwiseXor"
                ),
            ],
            inputs=[
                onnx.helper.make_value_info(
                    name="input1",
                    type_proto=onnx.helper.make_tensor_type_proto(
                        elem_type=onnx.TensorProto.INT32, shape=[1, 4]
                    ),
                ),
                onnx.helper.make_value_info(
                    name="input2",
                    type_proto=onnx.helper.make_tensor_type_proto(
                        elem_type=onnx.TensorProto.INT32, shape=[1, 4]
                    ),
                ),
            ],
            outputs=[
                onnx.helper.make_value_info(
                    name="output",
                    type_proto=onnx.helper.make_tensor_type_proto(
                        elem_type=onnx.TensorProto.INT32, shape=[1, 4]
                    ),
                )
            ]
        ),
    )


def main():
    onnx_model = build_model()
    file_name = "bitwise_xor.onnx"
    
    onnx.checker.check_model(onnx_model)  # Ensure valid ONNX
    onnx.save(onnx_model, file_name)  # Save the model
    print(f"Finished exporting model to {file_name}")

if __name__ == "__main__":
    main()