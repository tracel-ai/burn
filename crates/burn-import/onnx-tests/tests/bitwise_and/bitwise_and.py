#!/usr/bin/env python3
# used to generate all bitwise_and ONNX models

import onnx

def build_model(name, input1_shape, input2_shape, output_shape):
    """
    Build a BitwiseAnd ONNX model with specified input/output shapes.
    
    Args:
        name: Name of the model (used for file naming)
        input1_shape: Shape of first input ([] for scalar)
        input2_shape: Shape of second input ([] for scalar)
        output_shape: Shape of output ([] for scalar)
    """
    op_type = "BitwiseAnd"
    
    nodes = [
        onnx.helper.make_node(
            op_type,
            inputs=["input1", "input2"],
            outputs=["output"],
            name=f"/{op_type}"
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
        # (name, input1_shape, input2_shape, output_shape)
        ("bitwise_and", [4], [4], [4]),
        ("bitwise_and_scalar", [4], [], [4]),
        ("scalar_bitwise_and", [], [4], [4]),
        ("scalar_bitwise_and_scalar", [], [], []),
    ]
    
    for config in configs:
        build_model(*config)