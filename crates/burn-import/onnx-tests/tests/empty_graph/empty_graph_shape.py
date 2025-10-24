#!/usr/bin/env python3

import onnx
from onnx import TensorProto, helper

def create_model():
    # Create a model that returns a shape input directly
    # forward(a) { return a } where a is a 1D array like [2, 3, 4]
    # Empty graph with no nodes - input directly becomes output

    # Create input - 1D int64 array (shape)
    input_shape = helper.make_tensor_value_info(
        'input', TensorProto.INT64, [3]  # 3-element array
    )

    # Create output - same as input (direct passthrough)
    output = helper.make_tensor_value_info(
        'input', TensorProto.INT64, [3]  # Use same name as input for direct passthrough
    )

    # Create empty graph (no nodes)
    graph = helper.make_graph(
        [],  # No nodes
        'empty_graph_shape',
        [input_shape],
        [output]
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
    onnx.save(model, 'empty_graph_shape.onnx')
    print("Created empty_graph_shape.onnx")
