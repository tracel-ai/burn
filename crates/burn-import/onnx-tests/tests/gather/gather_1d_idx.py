#!/usr/bin/env python3

# used to generate model: onnx-tests/tests/gather/gather.onnx

# There is no current support for `Split`, and the `for` loop over the indices
# results in a `Split` node in the ONNX model.
# Therefore, this model is built and exported using ONNX directly.

import onnx


def build_model():
    return onnx.helper.make_model(
        ir_version=8,
        opset_imports=[onnx.helper.make_operatorsetid("", 16)],
        graph=onnx.helper.make_graph(name="main_graph", nodes=[
            onnx.helper.make_node(
                "Gather",
                inputs=["input1", "input2"],
                outputs=["output1"],
                name="/Gather",
                axis=1
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
                    elem_type=onnx.TensorProto.INT64, shape=[2]
                ),
            ),

        ],
        outputs=[
            onnx.helper.make_value_info(
                name="output1",
                type_proto=onnx.helper.make_tensor_type_proto(
                    elem_type=onnx.TensorProto.FLOAT, shape=[2, 2]
                ),
            )
        ]),
    )


def main():
    onnx_model = build_model()
    file_name = "gather_1d_idx.onnx"

    # Ensure valid ONNX:
    onnx.checker.check_model(onnx_model)

    onnx.save(onnx_model, file_name)


if __name__ == '__main__':
    main()
