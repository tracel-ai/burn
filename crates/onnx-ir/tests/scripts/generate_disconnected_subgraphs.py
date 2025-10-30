#!/usr/bin/env -S uv run
# /// script
# dependencies = [
#   "onnx>=1.15.0",
#   "numpy>=1.24.0",
# ]
# ///

"""
Generate ONNX model with multiple disconnected subgraphs.

Tests:
- Type inference with disconnected computation paths
- Edge case #27: Multiple disconnected subgraphs
"""

import onnx
from onnx import helper, TensorProto


def create_disconnected_subgraphs_model():
    """Create model with multiple independent computation paths."""

    # Two independent input groups
    input1 = helper.make_tensor_value_info('input1', TensorProto.FLOAT, [2, 3])
    input2 = helper.make_tensor_value_info('input2', TensorProto.FLOAT, [2, 3])
    input3 = helper.make_tensor_value_info('input3', TensorProto.FLOAT, [4, 5])
    input4 = helper.make_tensor_value_info('input4', TensorProto.FLOAT, [4, 5])

    # Two independent output groups
    output1 = helper.make_tensor_value_info('output1', TensorProto.FLOAT, [2, 3])
    output2 = helper.make_tensor_value_info('output2', TensorProto.FLOAT, [4, 5])

    # Subgraph 1: input1, input2 → output1 (completely independent)
    # Subgraph 2: input3, input4 → output2 (completely independent)
    nodes = [
        # Subgraph 1
        helper.make_node('Add', ['input1', 'input2'], ['temp1'], name='add_subgraph1'),
        helper.make_node('Relu', ['temp1'], ['output1'], name='relu_subgraph1'),

        # Subgraph 2 (completely disconnected from subgraph 1)
        helper.make_node('Mul', ['input3', 'input4'], ['temp2'], name='mul_subgraph2'),
        helper.make_node('Abs', ['temp2'], ['output2'], name='abs_subgraph2'),
    ]

    # Create the graph
    graph = helper.make_graph(
        nodes,
        'disconnected_subgraphs_model',
        [input1, input2, input3, input4],
        [output1, output2],
    )

    # Create the model
    model = helper.make_model(graph, producer_name="onnx-ir-test", opset_imports=[helper.make_opsetid("", 16)])

    # Check the model
    onnx.checker.check_model(model)

    return model


def main():
    """Generate and save the ONNX model."""
    model = create_disconnected_subgraphs_model()

    # Save the model
    output_path = '../fixtures/disconnected_subgraphs.onnx'
    onnx.save(model, output_path)
    print(f"Model saved to {output_path}")

    print(f"\nModel info:")
    print(f"  Two completely independent subgraphs")
    print(f"  Subgraph 1: input1, input2 → output1")
    print(f"  Subgraph 2: input3, input4 → output2")
    print(f"  Tests type inference with disconnected paths")


if __name__ == '__main__':
    main()
