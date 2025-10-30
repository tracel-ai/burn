#!/usr/bin/env -S uv run
# /// script
# dependencies = [
#   "onnx>=1.15.0",
#   "numpy>=1.24.0",
# ]
# ///

"""
Generate ONNX model with Shape type in broadcasting context.

Tests:
- Broadcasting with Shape argument type
- Edge case #30: Shape type in broadcasting context
"""

import onnx
from onnx import helper, TensorProto


def create_shape_broadcasting_model():
    """Create model where Shape output is used in broadcasting operation."""

    # Input
    input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, [2, 3, 4])

    # Output
    output = helper.make_tensor_value_info('output', TensorProto.INT64, [3])

    # Get shape, then use it in operations
    nodes = [
        # Extract shape as int64 tensor [2, 3, 4]
        helper.make_node('Shape', ['input'], ['shape_tensor'], name='shape'),

        # Use shape in Add operation (shape + shape, element-wise)
        # This tests Shape type in broadcasting context
        helper.make_node('Add', ['shape_tensor', 'shape_tensor'], ['output'], name='add_shapes'),
    ]

    # Create the graph
    graph = helper.make_graph(
        nodes,
        'shape_broadcasting_model',
        [input_tensor],
        [output],
    )

    # Create the model
    model = helper.make_model(graph, producer_name="onnx-ir-test", opset_imports=[helper.make_opsetid("", 16)])

    # Check the model
    onnx.checker.check_model(model)

    return model


def main():
    """Generate and save the ONNX model."""
    model = create_shape_broadcasting_model()

    # Save the model
    output_path = '../fixtures/shape_broadcasting.onnx'
    onnx.save(model, output_path)
    print(f"Model saved to {output_path}")

    print(f"\nModel info:")
    print(f"  Shape operation produces shape tensor")
    print(f"  Shape tensor used in Add (broadcasting context)")
    print(f"  Tests ArgType::Shape in broadcasting operations")


if __name__ == '__main__':
    main()
