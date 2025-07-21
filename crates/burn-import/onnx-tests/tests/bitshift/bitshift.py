#!/usr/bin/env python3
# used to generate model: onnx-tests/tests/bitshift/bitshift_left.onnx and bitshift_right.onnx

import onnx

def build_model(direction: str = "LEFT", scalar_shift: bool = False, scalar_first: bool = False):
    op_type = "BitShift"
    direction_attr = "LEFT" if direction == "LEFT" else "RIGHT"
    
    nodes = [
        onnx.helper.make_node(
            op_type,
            inputs=["x", "shift"],
            outputs=["output"],
            name=f"/{op_type}",
            direction=direction_attr
        ),
    ]
    
    # Determine input shapes based on scalar position
    if scalar_first:
        x_shape = []
        shift_shape = [4]
    else:
        x_shape = [4]
        shift_shape = [] if scalar_shift else [4]
    
    inputs = [
        onnx.helper.make_value_info(
            name="x",
            type_proto=onnx.helper.make_tensor_type_proto(
                elem_type=onnx.TensorProto.INT32, shape=x_shape
            ),
        ),
        onnx.helper.make_value_info(
            name="shift",
            type_proto=onnx.helper.make_tensor_type_proto(
                elem_type=onnx.TensorProto.INT32, shape=shift_shape
            ),
        ),
    ]
    
    return onnx.helper.make_model(
        ir_version=8,
        opset_imports=[onnx.helper.make_operatorsetid("", 18)],
        graph=onnx.helper.make_graph(
            name="main_graph",
            nodes=nodes,
            inputs=inputs,
            outputs=[
                onnx.helper.make_value_info(
                    name="output",
                    type_proto=onnx.helper.make_tensor_type_proto(
                        elem_type=onnx.TensorProto.INT32, shape=[4]
                    ),
                )
            ],
            initializer=[]
        ),
    )

def export_bitshift(direction: str = "LEFT"):
    # Regular tensor version
    onnx_model = build_model(direction, scalar_shift=False)
    file_name = f"bitshift_{direction.lower()}.onnx"
    
    onnx.checker.check_model(onnx_model)
    onnx.save(onnx_model, file_name)
    print(f"Finished exporting model to {file_name}")
    
    # Tensor-Scalar version
    onnx_model_scalar = build_model(direction, scalar_shift=True)
    file_name_scalar = f"bitshift_{direction.lower()}_scalar.onnx"
    
    onnx.checker.check_model(onnx_model_scalar)
    onnx.save(onnx_model_scalar, file_name_scalar)
    print(f"Finished exporting model to {file_name_scalar}")
    
    # Scalar-Tensor version
    onnx_model_scalar_first = build_model(direction, scalar_first=True)
    file_name_scalar_first = f"scalar_bitshift_{direction.lower()}.onnx"
    
    onnx.checker.check_model(onnx_model_scalar_first)
    onnx.save(onnx_model_scalar_first, file_name_scalar_first)
    print(f"Finished exporting scalar-first model to {file_name_scalar_first}")

if __name__ == "__main__":
    for direction in ["LEFT", "RIGHT"]:
        export_bitshift(direction)