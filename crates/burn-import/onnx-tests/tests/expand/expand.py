#!/usr/bin/env python3

# used to generate model: onnx-tests/tests/expand/expand.onnx

import onnx
from onnx import helper, TensorProto

def main() -> None:
    # Create a constant node for the input tensor
    input_node: onnx.NodeProto = helper.make_node(
        'Constant',
        inputs=[],
        outputs=['input_tensor'],
        value=helper.make_tensor(
            name='const_input',
            data_type=TensorProto.FLOAT,
            dims=[2, 1],
            vals=[1.0, 2.0]
        )
    )

    # Create a constant node for the shape tensor which specifies the expansion
    shape_node: onnx.NodeProto = helper.make_node(
        'Constant',
        inputs=[],
        outputs=['shape_tensor'],
        value=helper.make_tensor(
            name='const_shape',
            data_type=TensorProto.INT64,
            dims=[2],
            vals=[2, 2]  # Expanding each dimension to have 2 elements
        )
    )

    # Define the Expand node that uses the outputs from the constant nodes
    expand_node: onnx.NodeProto = helper.make_node(
        'Expand',
        inputs=['input_tensor', 'shape_tensor'],
        outputs=['output']
    )

    # Create the graph
    graph_def: onnx.GraphProto = helper.make_graph(
        nodes=[input_node, shape_node, expand_node],
        name='ExpandGraph',
        inputs=[],  # No inputs since all are provided by constants within the graph
        outputs=[
            helper.make_tensor_value_info('output', TensorProto.FLOAT, [2, 2])
        ]
    )

    # Create the model
    model_def: onnx.ModelProto = helper.make_model(graph_def, producer_name='expand')

    # Save the model to a file
    onnx.save(model_def, 'expand.onnx')

if __name__ == '__main__':
    main()
