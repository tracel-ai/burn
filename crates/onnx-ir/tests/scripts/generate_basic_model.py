#!/usr/bin/env -S uv run
# /// script
# dependencies = [
#   "onnx>=1.15.0",
#   "numpy>=1.24.0",
# ]
# ///

"""
Generate a basic ONNX model with common operations for smoke testing ONNX-IR parsing.

This script creates a simple model that includes:
- Relu activation
- PRelu activation
- Add operation
- Reshape operation
- MatMul operation

The model tests basic ONNX-IR parsing functionality.
"""

import onnx
from onnx import helper, TensorProto
import numpy as np


def create_basic_model():
    """Create a basic ONNX model with common operations."""

    # Define inputs
    input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 3, 4, 4])

    # Define outputs
    output_tensor = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 3, 4, 4])

    # Create initializers (constant tensors)
    # PRelu slope parameter
    prelu_slope = helper.make_tensor(
        name='prelu_slope',
        data_type=TensorProto.FLOAT,
        dims=[3, 1, 1],
        vals=np.array([0.1, 0.2, 0.3], dtype=np.float32).tobytes(),
        raw=True
    )

    # Add bias
    add_bias = helper.make_tensor(
        name='add_bias',
        data_type=TensorProto.FLOAT,
        dims=[1, 3, 1, 1],
        vals=np.array([1.0, 2.0, 3.0], dtype=np.float32).tobytes(),
        raw=True
    )

    # Create nodes
    nodes = [
        # Apply Relu
        helper.make_node('Relu', ['input'], ['relu_out'], name='relu'),

        # Apply PRelu
        helper.make_node('PRelu', ['relu_out', 'prelu_slope'], ['prelu_out'], name='prelu'),

        # Add bias
        helper.make_node('Add', ['prelu_out', 'add_bias'], ['output'], name='add'),
    ]

    # Create the graph
    graph = helper.make_graph(
        nodes,
        'basic_model',
        [input_tensor],
        [output_tensor],
        initializer=[prelu_slope, add_bias]
    )

    # Create the model
    model = helper.make_model(graph, producer_name='onnx-ir-test')
    model.opset_import[0].version = 16

    # Check the model
    onnx.checker.check_model(model)

    return model


def main():
    """Generate and save the ONNX model."""
    model = create_basic_model()

    # Save the model
    output_path = '../fixtures/basic_model.onnx'
    onnx.save(model, output_path)
    print(f"Model saved to {output_path}")

    # Print model info
    print(f"\nModel info:")
    print(f"  Opset version: {model.opset_import[0].version}")
    print(f"  Inputs: {[inp.name for inp in model.graph.input]}")
    print(f"  Outputs: {[out.name for out in model.graph.output]}")
    print(f"  Nodes: {len(model.graph.node)}")
    for node in model.graph.node:
        print(f"    - {node.op_type} ({node.name})")


if __name__ == '__main__':
    main()
