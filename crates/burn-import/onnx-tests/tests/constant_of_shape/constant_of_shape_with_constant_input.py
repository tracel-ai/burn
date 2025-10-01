#!/usr/bin/env python3

# used to generate model: constant_of_shape_with_constant_input.onnx

# This test creates a ConstantOfShape where the input shape comes from a Constant node
# This tests the case where the constant needs to be lifted to get the shape information

import onnx
import onnx.helper
import numpy as np


def build_model():
    # Create a constant node that outputs the shape [2, 3, 4]
    shape_constant = onnx.helper.make_node(
        "Constant",
        inputs=[],
        outputs=["shape_tensor"],
        name="constant_shape",
        value=onnx.helper.make_tensor(
            "shape_value",
            data_type=onnx.TensorProto.INT64,
            dims=[3],  # 1D tensor with 3 elements
            vals=[2, 3, 4]  # The shape values
        )
    )
    
    # Create ConstantOfShape that uses the constant as input
    constant_of_shape = onnx.helper.make_node(
        "ConstantOfShape",
        inputs=["shape_tensor"],
        outputs=["output1"],
        name="constantofshape1",
        value=onnx.helper.make_tensor(
            "value",
            data_type=onnx.TensorProto.INT64,
            dims=[1],
            vals=[1]  # Fill value
        )
    )
    
    # Build the graph
    graph = onnx.helper.make_graph(
        name="main_graph",
        nodes=[shape_constant, constant_of_shape],
        inputs=[],  # No inputs - fully constant
        outputs=[
            onnx.helper.make_value_info(
                name="output1",
                type_proto=onnx.helper.make_tensor_type_proto(
                    elem_type=onnx.TensorProto.INT64,
                    shape=[2, 3, 4]  # Expected output shape
                ),
            )
        ],
        initializer=[],
        value_info=[]
    )
    
    # Create the model
    model = onnx.helper.make_model(
        graph=graph,
        ir_version=8,
        opset_imports=[onnx.helper.make_operatorsetid("", 16)],
    )
    
    return model


def main():
    onnx_model = build_model()
    file_name = "constant_of_shape_with_constant_input.onnx"
    onnx.save(onnx_model, file_name)
    onnx.checker.check_model(file_name)
    print(f"Model saved to {file_name}")


if __name__ == "__main__":
    main()