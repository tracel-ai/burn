#!/usr/bin/env python3

import onnx
from onnx import TensorProto, helper

def create_model():
    # Create a model that returns multiple inputs directly
    # forward(a, b) { return (a, b) }
    # Tests: shape array and tensor
    # Empty graph with no nodes - inputs directly become outputs

    # Create inputs
    input1 = helper.make_tensor_value_info(
        'input1', TensorProto.INT64, [3]  # 1D array (shape)
    )
    input2 = helper.make_tensor_value_info(
        'input2', TensorProto.FLOAT, [2, 3]  # 2D tensor
    )

    # Create outputs (same names as inputs for direct passthrough)
    output1 = helper.make_tensor_value_info(
        'input1', TensorProto.INT64, [3]  # Use same name as input1
    )
    output2 = helper.make_tensor_value_info(
        'input2', TensorProto.FLOAT, [2, 3]  # Use same name as input2
    )

    # Create empty graph (no nodes)
    graph = helper.make_graph(
        [],  # No nodes
        'empty_graph_multiple',
        [input1, input2],
        [output1, output2]
    )

    # Create the model
    model = helper.make_model(graph)
    model.opset_import[0].version = 16

    return model

if __name__ == '__main__':
    model = create_model()

    # Validate
    onnx.checker.check_model(model)

    # Save
    onnx.save(model, 'empty_graph_multiple.onnx')
    print("Created empty_graph_multiple.onnx")
