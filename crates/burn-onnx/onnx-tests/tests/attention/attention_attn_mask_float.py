#!/usr/bin/env python3

# used to generate model: attention_attn_mask_float.onnx

import numpy as np
import onnx
import onnx.helper
from onnx import TensorProto
from onnx.reference import ReferenceEvaluator


def build_model():
    # Define the graph inputs and outputs
    q = onnx.helper.make_tensor_value_info("q", TensorProto.FLOAT, [1, 1, 2, 2])
    k = onnx.helper.make_tensor_value_info("k", TensorProto.FLOAT, [1, 1, 2, 2])
    v = onnx.helper.make_tensor_value_info("v", TensorProto.FLOAT, [1, 1, 2, 2])
    attn_mask = onnx.helper.make_tensor_value_info(
        "attn_mask", TensorProto.FLOAT, [2, 2]
    )
    y = onnx.helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1, 2, 2])

    # Create the GroupNormalization node
    attention = onnx.helper.make_node(
        "Attention",
        inputs=["q", "k", "v", "attn_mask"],
        outputs=["y"],
        name="AttentionNode",
    )

    # Create the graph
    graph = onnx.helper.make_graph(
        [attention],
        "AttentionModel",
        [q, k, v, attn_mask],
        [y],
    )

    # Create the model
    model = onnx.helper.make_model(
        opset_imports=[onnx.helper.make_operatorsetid("", 23)],
        graph=graph,
        producer_name="ONNX_Generator",
    )

    return model


if __name__ == "__main__":
    # Set seed and precision
    np.random.seed(42)
    np.set_printoptions(precision=8)

    # Build model
    q = np.array([[[[1.0, 0.0], [0.0, 1.0]]]])
    k = np.array([[[[0.0, 1.0], [1.0, 0.0]]]])
    v = np.array([[[[0.25, 0.5], [0.3, 0.6]]]])
    attn_mask = np.array([[2.0, 0.0], [0.0, 3.0]])
    onnx_model = build_model()
    file_name = "attention_attn_mask_float.onnx"

    # Ensure valid ONNX and save
    onnx.checker.check_model(onnx_model)
    onnx.save(onnx_model, file_name)
    print(f"Finished exporting model to {file_name}")

    # Output some test data for use in the test
    print(f"Test input data: {repr(q)} {repr(k)} {repr(v)} {repr(attn_mask)}")
    print(f"Test input data shape: {q.shape} {k.shape} {v.shape} {attn_mask.shape}")
    session = ReferenceEvaluator("attention_attn_mask_float.onnx", verbose=1)
    (test_output,) = session.run(None, {"q": q, "k": k, "v": v, "attn_mask": attn_mask})
    print(f"Test output: {repr(test_output)}")
    print(f"Test output shape: {test_output.shape}")
