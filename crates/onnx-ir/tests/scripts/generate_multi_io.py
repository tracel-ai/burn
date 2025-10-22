#!/usr/bin/env -S uv run
# /// script
# dependencies = [
#   "onnx>=1.15.0",
#   "numpy>=1.24.0",
# ]
# ///

"""
Generate ONNX model with multiple inputs and outputs.

Tests:
- Multiple graph inputs (3 inputs)
- Multiple graph outputs (2 outputs)
- Input name mapping and resolution
- Proper output routing
"""

import onnx
from onnx import helper, TensorProto
import numpy as np


def create_multi_io_model():
    """Create model with 3 inputs and 2 outputs."""

    # Define 3 inputs
    input1 = helper.make_tensor_value_info('input_a', TensorProto.FLOAT, [1, 4])
    input2 = helper.make_tensor_value_info('input_b', TensorProto.FLOAT, [1, 4])
    input3 = helper.make_tensor_value_info('input_c', TensorProto.FLOAT, [1, 4])

    # Define 2 outputs
    output1 = helper.make_tensor_value_info('output_sum', TensorProto.FLOAT, [1, 4])
    output2 = helper.make_tensor_value_info('output_product', TensorProto.FLOAT, [1, 4])

    # Create nodes
    nodes = [
        # Add input_a + input_b
        helper.make_node('Add', ['input_a', 'input_b'], ['sum_ab'], name='add1'),

        # Multiply sum_ab * input_c
        helper.make_node('Mul', ['sum_ab', 'input_c'], ['output_product'], name='mul1'),

        # Add sum_ab + input_c for second output
        helper.make_node('Add', ['sum_ab', 'input_c'], ['output_sum'], name='add2'),
    ]

    # Create the graph
    graph = helper.make_graph(
        nodes,
        'multi_io_model',
        [input1, input2, input3],
        [output1, output2],
    )

    # Create the model
    model = helper.make_model(graph, producer_name='onnx-ir-test')
    model.opset_import[0].version = 16

    # Check the model
    onnx.checker.check_model(model)

    return model


def main():
    """Generate and save the ONNX model."""
    model = create_multi_io_model()

    # Save the model
    output_path = '../fixtures/multi_io.onnx'
    onnx.save(model, output_path)
    print(f"Model saved to {output_path}")

    # Print model info
    print(f"\nModel info:")
    print(f"  Opset version: {model.opset_import[0].version}")
    print(f"  Inputs: {[inp.name for inp in model.graph.input]}")
    print(f"  Outputs: {[out.name for out in model.graph.output]}")
    print(f"  Nodes: {len(model.graph.node)}")
    for node in model.graph.node:
        print(f"    - {node.op_type} ({node.name}): {list(node.input)} → {list(node.output)}")


if __name__ == '__main__':
    main()
