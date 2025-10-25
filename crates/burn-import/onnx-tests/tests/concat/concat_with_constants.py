#!/usr/bin/env python3

import numpy as np
import onnx
from onnx import helper, TensorProto

def main():
    # Create a graph that tests concat with constant rank-1 tensors
    # This simulates the scenario from xfeat model

    # Input tensor
    input_tensor = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [3, 4]
    )

    # Shape node to get shape of input
    shape_node = helper.make_node(
        "Shape",
        inputs=["input"],
        outputs=["input_shape"]
    )

    # Constant nodes with different dimensions
    const1_node = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["const1"],
        value=helper.make_tensor(
            name="const1_value",
            data_type=TensorProto.INT64,
            dims=[2],
            vals=[2, 3]
        )
    )

    const2_node = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["const2"],
        value=helper.make_tensor(
            name="const2_value",
            data_type=TensorProto.INT64,
            dims=[1],
            vals=[5]
        )
    )

    const3_node = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["const3"],
        value=helper.make_tensor(
            name="const3_value",
            data_type=TensorProto.INT64,
            dims=[3],
            vals=[7, 8, 9]
        )
    )

    # Concat node
    concat_node = helper.make_node(
        "Concat",
        inputs=["input_shape", "const1", "const2", "const3"],
        outputs=["output"],
        axis=0
    )

    # Output tensor
    output_tensor = helper.make_tensor_value_info(
        "output", TensorProto.INT64, [8]  # 2 + 2 + 1 + 3 = 8 elements
    )

    # Create graph
    graph = helper.make_graph(
        [shape_node, const1_node, const2_node, const3_node, concat_node],
        "concat_with_constants",
        [input_tensor],
        [output_tensor]
    )

    # Create model
    model = helper.make_model(graph,
            producer_name="concat_with_constants_test",
            opset_imports=[onnx.helper.make_operatorsetid("", 16)])

    # Save model
    onnx.save(model, "concat_with_constants.onnx")
    print("Model exported to concat_with_constants.onnx")

if __name__ == "__main__":
    main()
