#!/usr/bin/env python3
# used to generate all bitshift ONNX models

import onnx

def build_model(name, input1_shape, input2_shape, output_shape, direction):
    """
    Build a BitShift ONNX model with specified input/output shapes and direction.
    
    Args:
        name: Name of the model (used for file naming)
        input1_shape: Shape of first input ([] for scalar)
        input2_shape: Shape of second input ([] for scalar)
        output_shape: Shape of output ([] for scalar)
        direction: "LEFT" or "RIGHT"
    """
    op_type = "BitShift"
    
    nodes = [
        onnx.helper.make_node(
            op_type,
            inputs=["input1", "input2"],
            outputs=["output"],
            name=f"/{op_type}",
            direction=direction
        ),
    ]
    
    inputs = [
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
    ]
    
    outputs = [
        onnx.helper.make_value_info(
            name="output",
            type_proto=onnx.helper.make_tensor_type_proto(
                elem_type=onnx.TensorProto.INT32, shape=output_shape
            ),
        )
    ]
    
    model = onnx.helper.make_model(
        ir_version=8,
        opset_imports=[onnx.helper.make_operatorsetid("", 18)],
        graph=onnx.helper.make_graph(
            name="main_graph",
            nodes=nodes,
            inputs=inputs,
            outputs=outputs,
            initializer=[]
        ),
    )
    
    onnx.checker.check_model(model)
    onnx.save(model, f"{name}.onnx")
    print(f"Finished exporting model to {name}.onnx")

if __name__ == "__main__":
    # Define all model configurations
    configs = [
        # (name, input1_shape, input2_shape, output_shape, direction)
        ("bitshift_left", [4], [4], [4], "LEFT"),
        ("bitshift_right", [4], [4], [4], "RIGHT"),
        ("bitshift_left_scalar", [4], [], [4], "LEFT"),
        ("bitshift_right_scalar", [4], [], [4], "RIGHT"),
        ("scalar_bitshift_left", [], [4], [4], "LEFT"),
        ("scalar_bitshift_right", [], [4], [4], "RIGHT"),
        ("scalar_bitshift_left_scalar", [], [], [], "LEFT"),
        ("scalar_bitshift_right_scalar", [], [], [], "RIGHT"),
    ]
    
    for config in configs:
        build_model(*config)