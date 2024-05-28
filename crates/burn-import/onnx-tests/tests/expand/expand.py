#!/usr/bin/env python3

# used to generate model: onnx-tests/tests/expand/expand.onnx

import onnx
from onnx import helper, TensorProto

def main() -> None:

    # Define the Expand node that uses the outputs from the constant nodes
    expand_node: onnx.NodeProto = helper.make_node(
        'Expand',
        inputs=['input_tensor', 'shape_tensor'],
        outputs=['output']
    )

    # Create the graph
    graph_def: onnx.GraphProto = helper.make_graph(
        nodes=[expand_node],
        name='ExpandGraph',
        inputs=[
            helper.make_tensor_value_info('input_tensor', TensorProto.FLOAT, [1]),
            helper.make_tensor_value_info('shape_tensor', TensorProto.INT64, [1]),
            ],  # No inputs since all are provided by constants within the graph
        outputs=[
            helper.make_tensor_value_info('output', TensorProto.FLOAT, [2, 2])
        ],
    )

    # Create the model
    model_def: onnx.ModelProto = helper.make_model(graph_def, producer_name='expand')

    # Save the model to a file
    onnx.save(model_def, 'expand.onnx')

if __name__ == '__main__':
    main()
