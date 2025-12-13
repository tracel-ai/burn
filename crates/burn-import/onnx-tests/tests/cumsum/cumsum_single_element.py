#!/usr/bin/env python3

# used to generate model: onnx-tests/tests/cumsum/cumsum_single_element.onnx
# Edge case: single element tensor

import numpy as np
import onnx
from onnx import TensorProto, helper
from onnx.reference import ReferenceEvaluator


def main():
    # Create input tensor (single element)
    data = np.array([42.0], dtype=np.float32)

    # Create input tensor info
    data_tensor = helper.make_tensor_value_info(
        "data", TensorProto.FLOAT, list(data.shape)
    )

    # Create axis as initializer (constant)
    axis_tensor = helper.make_tensor("axis", TensorProto.INT64, [], [0])

    # Create output tensor info
    output_tensor = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, list(data.shape)
    )

    # Create CumSum node (default: exclusive=0, reverse=0)
    cumsum_node = helper.make_node(
        "CumSum", inputs=["data", "axis"], outputs=["output"], exclusive=0, reverse=0
    )

    # Create graph and model
    graph = helper.make_graph(
        [cumsum_node],
        "cumsum-single-element-model",
        [data_tensor],
        [output_tensor],
        initializer=[axis_tensor],
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 16)])
    onnx.checker.check_model(model)
    onnx.save(model, "cumsum_single_element.onnx")

    # Use ReferenceEvaluator for expected output (spec compliant)
    ref = ReferenceEvaluator(model)
    output = ref.run(None, {"data": data})[0]

    print("=== Values for mod.rs ===")
    print(f"Input data: {data.tolist()}")
    print(f"Output: {output.tolist()}")
    print("// Input: [42.]")
    print("// Expected output: [42.]")
    print("ONNX model 'cumsum_single_element.onnx' generated successfully.")


if __name__ == "__main__":
    main()
