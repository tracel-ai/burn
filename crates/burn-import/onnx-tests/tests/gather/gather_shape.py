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
            # First Shape node
            onnx.helper.make_node(
                "Shape",
                inputs=["input1"],
                outputs=["shape1"],
                name="/Shape"
            ),
            # Gather with runtime indices (from input2)
            onnx.helper.make_node(
                "Gather",
                inputs=["shape1", "input2"],
                outputs=["output1"],
                name="/Gather",
                axis=0
            ),
            # Constant node for scalar index
            onnx.helper.make_node(
                "Constant",
                inputs=[],
                outputs=["const_idx"],
                name="/Constant",
                value=onnx.helper.make_tensor(
                    name="const_value",
                    data_type=onnx.TensorProto.INT64,
                    dims=[],  # Scalar
                    vals=[1]
                )
            ),
            # Gather with constant scalar index
            onnx.helper.make_node(
                "Gather",
                inputs=["shape1", "const_idx"],
                outputs=["output2"],
                name="/Gather2",
                axis=0
            ),
            # Constant node for 1D indices
            onnx.helper.make_node(
                "Constant",
                inputs=[],
                outputs=["const_indices"],
                name="/Constant2",
                value=onnx.helper.make_tensor(
                    name="const_indices_value",
                    data_type=onnx.TensorProto.INT64,
                    dims=[2],  # 1D tensor with 2 elements
                    vals=[0, 1]
                )
            ),
            # Gather with constant 1D indices
            onnx.helper.make_node(
                "Gather",
                inputs=["shape1", "const_indices"],
                outputs=["output3"],
                name="/Gather3",
                axis=0
            ),
            # Constant node for negative indices
            onnx.helper.make_node(
                "Constant",
                inputs=[],
                outputs=["const_neg_indices"],
                name="/Constant3",
                value=onnx.helper.make_tensor(
                    name="const_neg_indices_value",
                    data_type=onnx.TensorProto.INT64,
                    dims=[2],  # 1D tensor with 2 elements
                    vals=[-1, -2]  # Last and second-to-last elements
                )
            ),
            # Gather with negative indices
            onnx.helper.make_node(
                "Gather",
                inputs=["shape1", "const_neg_indices"],
                outputs=["output4"],
                name="/Gather4",
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
            ),
            onnx.helper.make_value_info(
                name="output2",
                type_proto=onnx.helper.make_tensor_type_proto(
                    elem_type=onnx.TensorProto.INT64, shape=[]  # Scalar output
                ),
            ),
            onnx.helper.make_value_info(
                name="output3",
                type_proto=onnx.helper.make_tensor_type_proto(
                    elem_type=onnx.TensorProto.INT64, shape=[2]  # 1D tensor output with 2 elements
                ),
            ),
            onnx.helper.make_value_info(
                name="output4",
                type_proto=onnx.helper.make_tensor_type_proto(
                    elem_type=onnx.TensorProto.INT64, shape=[2]  # 1D tensor output with 2 elements (negative indices)
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
