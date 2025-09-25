#!/usr/bin/env python3

# Test: concat with Shape and single-element constant tensor

import onnx
import onnx.helper
import numpy as np


def build_model():
    # Create a constant node with a single-element rank-1 tensor
    const_node = onnx.helper.make_node(
        "Constant",
        inputs=[],
        outputs=["const_single"],
        value=onnx.helper.make_tensor(
            name="const_value",
            data_type=onnx.TensorProto.INT64,
            dims=[1],
            vals=[100]
        ),
        name="/Constant"
    )

    # Create Shape node to extract shape from input tensor
    shape_node = onnx.helper.make_node(
        "Shape",
        inputs=["input1"],
        outputs=["shape1"],
        name="/Shape"
    )

    # Create a Concat node that concatenates shape and single-element constant
    concat_node = onnx.helper.make_node(
        "Concat",
        inputs=["shape1", "const_single"],
        outputs=["concatenated"],
        axis=0,
        name="/Concat"
    )

    # Create the graph
    graph = onnx.helper.make_graph(
        name="main_graph",
        nodes=[const_node, shape_node, concat_node],
        inputs=[
            onnx.helper.make_value_info(
                name="input1",
                type_proto=onnx.helper.make_tensor_type_proto(
                    elem_type=onnx.TensorProto.FLOAT, shape=[2, 3]
                ),
            ),
        ],
        outputs=[
            onnx.helper.make_value_info(
                name="concatenated",
                type_proto=onnx.helper.make_tensor_type_proto(
                    elem_type=onnx.TensorProto.INT64, shape=[3]  # 2 + 1 = 3
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
    file_name = "concat_mixed_single_element.onnx"
    onnx.save(onnx_model, file_name)
    onnx.checker.check_model(file_name)

    print(f"Finished exporting model to {file_name}")

    # Test with onnx.reference.ReferenceEvaluator
    try:
        from onnx.reference import ReferenceEvaluator

        # Create test data
        test_input = np.ones((2, 3), dtype=np.float32)

        # Run inference
        sess = ReferenceEvaluator(onnx_model)
        result = sess.run(None, {"input1": test_input})

        print(f"Test input shape: {test_input.shape}")
        print(f"Concatenated output: {result[0]}")
        print(f"Expected: [2, 3, 100]")

    except ImportError:
        print("onnx.reference not available, skipping inference test")


if __name__ == "__main__":
    main()