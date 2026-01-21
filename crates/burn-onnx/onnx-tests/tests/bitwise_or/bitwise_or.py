import torch
import onnx


def build_model(scalar=False, scalar_first=False):
    if scalar_first:
        input1_shape = []
        input2_shape = [1, 4]
    else:
        input1_shape = [1, 4]
        input2_shape = [1, 4] if not scalar else []
    
    return onnx.helper.make_model(
        ir_version=8,
        opset_imports=[onnx.helper.make_operatorsetid("", 18)],
        graph=onnx.helper.make_graph(
            name="main_graph",
            nodes=[
                onnx.helper.make_node(
                    "BitwiseOr",
                    inputs=["input1", "input2"],
                    outputs=["output"],
                    name="/BitwiseOr"
                ),
            ],
            inputs=[
                onnx.helper.make_value_info(
                    name="input1",
                    type_proto=onnx.helper.make_tensor_type_proto(
                        elem_type=onnx.TensorProto.INT32, shape=input1_shape
                    ),
                ),
                onnx.helper.make_value_info(
                    name="input2",
                    type_proto=onnx.helper.make_tensor_type_proto(
                        elem_type=onnx.TensorProto.INT32, shape=input2_shape
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
    file_name = "bitwise_or.onnx"
    
    onnx.checker.check_model(onnx_model)  # Ensure valid ONNX
    onnx.save(onnx_model, file_name)  # Save the model
    print(f"Finished exporting model to {file_name}")
    
    onnx_scalar_model = build_model(scalar=True)
    scalar_file_name = "bitwise_or_scalar.onnx"
    
    onnx.checker.check_model(onnx_scalar_model)  # Ensure valid ONNX
    onnx.save(onnx_scalar_model, scalar_file_name)  # Save the model
    print(f"Finished exporting scalar model to {scalar_file_name}")
    
    # Scalar-Tensor version
    onnx_scalar_first_model = build_model(scalar_first=True)
    scalar_first_file_name = "scalar_bitwise_or.onnx"
    
    onnx.checker.check_model(onnx_scalar_first_model)  # Ensure valid ONNX
    onnx.save(onnx_scalar_first_model, scalar_first_file_name)  # Save the model
    print(f"Finished exporting scalar-first model to {scalar_first_file_name}")

if __name__ == "__main__":
    main()