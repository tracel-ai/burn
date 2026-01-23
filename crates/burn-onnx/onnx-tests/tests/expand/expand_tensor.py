#!/usr/bin/env python3

# used to generate model: onnx-tests/tests/expand/expand.onnx

import onnx
from onnx import helper, TensorProto

def main() -> None:
    # Define the Expand node that uses the outputs from the constant nodes
    expand_node = helper.make_node(
        'Expand',
        name='/Expand',
        inputs=['input_tensor', 'shape'],
        outputs=['output']
    )

    # Create the graph
    graph_def = helper.make_graph(
        nodes=[expand_node],
        name='ExpandGraph',
        inputs=[
            helper.make_tensor_value_info('input_tensor', TensorProto.FLOAT, [2, 1]),
            helper.make_tensor_value_info('shape', TensorProto.INT64, [2]),
        ],
        outputs=[
            helper.make_tensor_value_info('output', TensorProto.FLOAT, [2,2])
        ],
    )

    # Create the model
    model_def = helper.make_model(graph_def, producer_name='expand')

    
    # Ensure valid ONNX:
    onnx.checker.check_model(model_def)

    # Save the model to a file
    onnx.save(model_def, 'expand_tensor.onnx')

if __name__ == '__main__':
    main()
