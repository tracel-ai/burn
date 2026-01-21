#!/usr/bin/env python3

import onnx
from onnx import TensorProto, helper

def create_model():
    # Create a model that returns a scalar input directly
    # forward(a) { return a } where a is a scalar
    # Empty graph with no nodes - input directly becomes output

    # Create input - scalar float
    input_scalar = helper.make_tensor_value_info(
        'input', TensorProto.FLOAT, []  # Empty shape means scalar
    )

    # Create output - same as input (direct passthrough)
    output = helper.make_tensor_value_info(
        'input', TensorProto.FLOAT, []  # Use same name as input for direct passthrough
    )

    # Create empty graph (no nodes)
    graph = helper.make_graph(
        [],  # No nodes
        'empty_graph_scalar',
        [input_scalar],
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
    onnx.save(model, 'empty_graph_scalar.onnx')
    print("Created empty_graph_scalar.onnx")
