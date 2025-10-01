#!/usr/bin/env python3

# Test: concat with multiple Shapes and multiple constant tensors

import onnx
import onnx.helper
import numpy as np


def build_model():
    # Create first constant with 2 elements
    const1_node = onnx.helper.make_node(
        "Constant",
        inputs=[],
        outputs=["const1"],
        value=onnx.helper.make_tensor(
            name="const1_value",
            data_type=onnx.TensorProto.INT64,
            dims=[2],
            vals=[100, 200]
        ),
        name="/Constant1"
    )

    # Create second constant with 1 element
    const2_node = onnx.helper.make_node(
        "Constant",
        inputs=[],
        outputs=["const2"],
        value=onnx.helper.make_tensor(
            name="const2_value",
            data_type=onnx.TensorProto.INT64,
            dims=[1],
            vals=[300]
        ),
        name="/Constant2"
    )

    # Create Shape nodes for two input tensors
    shape1_node = onnx.helper.make_node(
        "Shape",
        inputs=["input1"],
        outputs=["shape1"],
        name="/Shape1"
    )

    shape2_node = onnx.helper.make_node(
        "Shape",
        inputs=["input2"],
        outputs=["shape2"],
        name="/Shape2"
    )

    # Create a Concat node with mixed inputs: shape, const, shape, const
    concat_node = onnx.helper.make_node(
        "Concat",
        inputs=["shape1", "const1", "shape2", "const2"],
        outputs=["concatenated"],
        axis=0,
        name="/Concat"
    )

    # Create the graph
    graph = onnx.helper.make_graph(
        name="main_graph",
        nodes=[const1_node, const2_node, shape1_node, shape2_node, concat_node],
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
                    elem_type=onnx.TensorProto.FLOAT, shape=[4, 5, 6]
                ),
            ),
        ],
        outputs=[
            onnx.helper.make_value_info(
                name="concatenated",
                type_proto=onnx.helper.make_tensor_type_proto(
                    elem_type=onnx.TensorProto.INT64, shape=[8]  # 2 + 2 + 3 + 1 = 8
                ),
            )
        ]
    )

    # Create the model
    model = onnx.helper.make_model(
        graph,
        ir_version=8,
        opset_imports=[onnx.helper.make_operatorsetid("", 16)]
    )

    return model


def main():
    onnx_model = build_model()
    file_name = "concat_multiple_mixed.onnx"
    onnx.save(onnx_model, file_name)
    onnx.checker.check_model(file_name)

    print(f"Finished exporting model to {file_name}")

    # Test with onnx.reference.ReferenceEvaluator
    try:
        from onnx.reference import ReferenceEvaluator

        # Create test data
        test_input1 = np.ones((2, 3), dtype=np.float32)
        test_input2 = np.ones((4, 5, 6), dtype=np.float32)

        # Run inference
        sess = ReferenceEvaluator(onnx_model)
        result = sess.run(None, {"input1": test_input1, "input2": test_input2})

        print(f"Test input1 shape: {test_input1.shape}")
        print(f"Test input2 shape: {test_input2.shape}")
        print(f"Concatenated output: {result[0]}")
        print(f"Expected: [2, 3, 100, 200, 4, 5, 6, 300]")

    except ImportError:
        print("onnx.reference not available, skipping inference test")


if __name__ == "__main__":
    main()