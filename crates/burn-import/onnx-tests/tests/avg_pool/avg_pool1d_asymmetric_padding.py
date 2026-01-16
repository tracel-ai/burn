#!/usr/bin/env python3

# used to generate model: avg_pool1d_asymmetric_padding.onnx

import numpy as np
import onnx
from onnx import helper, TensorProto
from onnx.reference import ReferenceEvaluator


def main():
    # Input: [batch=2, channels=4, width=10]
    # Asymmetric padding: left=1, right=2
    # kernel=3, stride=1
    # Output width = (10 + 1 + 2 - 3) / 1 + 1 = 11

    X = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 4, 10])
    Y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 4, 11])

    # Create AveragePool node with asymmetric padding (left=1, right=2)
    # ONNX pads format for 1D: [start, end] = [left, right]
    avg_pool_node = helper.make_node(
        "AveragePool",
        inputs=["x"],
        outputs=["y"],
        kernel_shape=[3],
        strides=[1],
        pads=[1, 2],  # [left, right] asymmetric padding
        count_include_pad=1,  # Include padding in average calculation
    )

    graph = helper.make_graph(
        [avg_pool_node],
        "avg_pool1d_asymmetric_padding",
        [X],
        [Y],
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 8

    onnx.checker.check_model(model)
    file_name = "avg_pool1d_asymmetric_padding.onnx"
    onnx.save(model, file_name)

    print("Finished exporting model to {}".format(file_name))
    print("Ops in graph: {}".format([n.op_type for n in model.graph.node]))

    # Verify with ReferenceEvaluator
    test_input = np.ones((2, 4, 10), dtype=np.float32)
    ref = ReferenceEvaluator(file_name)
    ref_output = ref.run(None, {"x": test_input})[0]

    print("Test input shape: {}".format(test_input.shape))
    print("Test output shape: {}".format(ref_output.shape))
    print("ReferenceEvaluator output sum: {}".format(ref_output.sum()))


if __name__ == "__main__":
    main()
