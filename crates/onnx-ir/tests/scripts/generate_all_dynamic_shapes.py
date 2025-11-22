#!/usr/bin/env -S uv run
# /// script
# dependencies = [
#   "onnx>=1.15.0",
#   "numpy>=1.24.0",
# ]
# ///

"""
Generate ONNX model with all dynamic shapes (no static shapes).

Tests:
- Type inference without static shape information
- Edge case #12: All dynamic shapes (no static)
"""

import onnx
from onnx import helper, TensorProto


def create_all_dynamic_shapes_model():
    """Create model where all shapes are dynamic (symbolic)."""

    # Dynamic shapes using symbolic dimensions
    input1 = helper.make_tensor_value_info('input1', TensorProto.FLOAT, ['batch', 'channels', 'height', 'width'])
    input2 = helper.make_tensor_value_info('input2', TensorProto.FLOAT, ['batch', 'channels', 'height', 'width'])

    # Output also has dynamic shape
    output = helper.make_tensor_value_info('output', TensorProto.FLOAT, ['batch', 'channels', 'height', 'width'])

    # Operations with dynamic shapes
    nodes = [
        # All operations work with fully dynamic shapes
        helper.make_node('Add', ['input1', 'input2'], ['temp1'], name='add'),
        helper.make_node('Relu', ['temp1'], ['temp2'], name='relu'),
        helper.make_node('Mul', ['temp2', 'input1'], ['output'], name='mul'),
    ]

    # Create the graph
    graph = helper.make_graph(
        nodes,
        'all_dynamic_shapes_model',
        [input1, input2],
        [output],
    )

    # Create the model
    model = helper.make_model(graph, producer_name="onnx-ir-test", opset_imports=[helper.make_opsetid("", 16)])

    # Check the model
    onnx.checker.check_model(model)

    return model


def main():
    """Generate and save the ONNX model."""
    model = create_all_dynamic_shapes_model()

    # Save the model
    output_path = '../fixtures/all_dynamic_shapes.onnx'
    onnx.save(model, output_path)
    print(f"Model saved to {output_path}")

    print(f"\nModel info:")
    print(f"  All shapes are symbolic/dynamic")
    print(f"  Tests type inference without static shape info")


if __name__ == '__main__':
    main()
