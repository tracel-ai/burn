#!/usr/bin/env python3

# used to generate model: maxpool2d_asymmetric_padding.onnx

import numpy as np
import onnx
from onnx import helper, TensorProto
from onnxruntime import InferenceSession


def main():
    # Input: [batch=2, channels=4, height=10, width=15]
    # Asymmetric padding: top=1, left=1, bottom=2, right=2 (pad must be < kernel for ONNX Runtime)
    # kernel=[3,3], stride=[1,1]
    # Output height = (10 + 1 + 2 - 3) / 1 + 1 = 11
    # Output width = (15 + 1 + 2 - 3) / 1 + 1 = 16

    X = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 4, 10, 15])
    Y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 4, 11, 16])

    # Create MaxPool node with asymmetric padding
    # ONNX pads format for 2D: [top, left, bottom, right]
    max_pool_node = helper.make_node(
        "MaxPool",
        inputs=["x"],
        outputs=["y"],
        kernel_shape=[3, 3],
        strides=[1, 1],
        pads=[1, 1, 2, 2],  # [top, left, bottom, right] asymmetric padding
    )

    graph = helper.make_graph(
        [max_pool_node],
        "maxpool2d_asymmetric_padding",
        [X],
        [Y],
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 8

    onnx.checker.check_model(model)
    file_name = "maxpool2d_asymmetric_padding.onnx"
    onnx.save(model, file_name)

    print("Finished exporting model to {}".format(file_name))
    print("Ops in graph: {}".format([n.op_type for n in model.graph.node]))

    # Verify with ONNX Runtime (ReferenceEvaluator has a bug with MaxPool asymmetric padding)
    test_input = np.ones((2, 4, 10, 15), dtype=np.float32)
    session = InferenceSession(file_name)
    ort_output = session.run(None, {"x": test_input})[0]

    print("Test input shape: {}".format(test_input.shape))
    print("Test output shape: {}".format(ort_output.shape))
    print("ONNX Runtime output sum: {}".format(ort_output.sum()))


if __name__ == "__main__":
    main()
