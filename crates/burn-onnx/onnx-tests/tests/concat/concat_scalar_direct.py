#!/usr/bin/env python3

# used to generate model: concat_scalar_direct.onnx
# This test directly reproduces issue #4228: Concat fails when receiving Scalar(I64) input.
#
# Pattern: Shape -> Gather (scalar index) -> Concat (direct, no Unsqueeze)
# This is the exact pattern that causes the TypeMismatch error in onnx-ir.

import onnx
import onnx.helper
import numpy as np


def build_model():
    # Get shape of input tensor: [batch, channels, height, width]
    shape_node = onnx.helper.make_node(
        "Shape",
        inputs=["input1"],
        outputs=["shape1"],
        name="/Shape"
    )

    # Constant scalar index (0) to extract batch dimension
    # Using shape=[] makes it a scalar - this is the key to the bug
    const_idx_node = onnx.helper.make_node(
        "Constant",
        inputs=[],
        outputs=["idx"],
        value=onnx.helper.make_tensor(
            name="idx_value",
            data_type=onnx.TensorProto.INT64,
            dims=[],  # Scalar
            vals=[0]
        ),
        name="/ConstIdx"
    )

    # Gather the batch dimension (index 0) from shape
    # With scalar index, output is also scalar (Scalar(I64) in onnx-ir)
    gather_node = onnx.helper.make_node(
        "Gather",
        inputs=["shape1", "idx"],
        outputs=["batch_dim"],  # This will be Scalar(I64) - triggers the bug
        axis=0,
        name="/Gather"
    )

    # Constant for another dimension
    const_dim_node = onnx.helper.make_node(
        "Constant",
        inputs=[],
        outputs=["other_dim"],
        value=onnx.helper.make_tensor(
            name="other_dim_value",
            data_type=onnx.TensorProto.INT64,
            dims=[],  # Also scalar
            vals=[64]
        ),
        name="/ConstDim"
    )

    # Concat two scalars directly - this triggers the bug
    # In ONNX, this is valid and produces a 1D tensor of length 2
    concat_node = onnx.helper.make_node(
        "Concat",
        inputs=["batch_dim", "other_dim"],
        outputs=["output_shape"],
        axis=0,
        name="/Concat"
    )

    # Create the graph
    graph = onnx.helper.make_graph(
        name="main_graph",
        nodes=[
            shape_node,
            const_idx_node,
            gather_node,
            const_dim_node,
            concat_node
        ],
        inputs=[
            onnx.helper.make_value_info(
                name="input1",
                type_proto=onnx.helper.make_tensor_type_proto(
                    elem_type=onnx.TensorProto.FLOAT, shape=[2, 3, 4, 5]
                ),
            ),
        ],
        outputs=[
            onnx.helper.make_value_info(
                name="output_shape",
                type_proto=onnx.helper.make_tensor_type_proto(
                    elem_type=onnx.TensorProto.INT64, shape=[2]  # [batch_dim, 64]
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
    file_name = "concat_scalar_direct.onnx"
    onnx.save(onnx_model, file_name)
    onnx.checker.check_model(file_name)

    print(f"Finished exporting model to {file_name}")

    # Test with onnx.reference.ReferenceEvaluator
    try:
        from onnx.reference import ReferenceEvaluator

        # Create test data with shape [2, 3, 4, 5]
        test_input = np.ones((2, 3, 4, 5), dtype=np.float32)

        # Run inference
        sess = ReferenceEvaluator(onnx_model)
        result = sess.run(None, {"input1": test_input})

        print(f"Test input shape: {test_input.shape}")
        print(f"Output shape tensor: {result[0]}")
        print(f"Expected: [2, 64] (batch=2 from input shape, 64 from constant)")

    except ImportError:
        print("onnx.reference not available, skipping inference test")


if __name__ == "__main__":
    main()
