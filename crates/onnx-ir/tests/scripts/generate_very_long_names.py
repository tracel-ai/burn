#!/usr/bin/env -S uv run
# /// script
# dependencies = [
#   "onnx>=1.15.0",
#   "numpy>=1.24.0",
# ]
# ///

"""
Generate ONNX model with very long node names (100+ characters).

Tests:
- String handling for very long names
- Edge case #40: Very long node names (100+ chars)
"""

import onnx
from onnx import helper, TensorProto


def create_very_long_names_model():
    """Create model with very long node and tensor names."""

    # Long names (120+ characters)
    long_input_name = 'input_with_extremely_long_name_that_exceeds_one_hundred_characters_to_test_string_handling_in_the_parser_and_ir_converter_xyz'
    long_output_name = 'output_with_extremely_long_name_that_exceeds_one_hundred_characters_to_test_string_handling_in_the_parser_and_ir_converter_abc'
    long_node_name = 'relu_node_with_extremely_long_name_that_exceeds_one_hundred_characters_to_test_string_handling_capabilities_qwerty'

    # Input and output with long names
    input_tensor = helper.make_tensor_value_info(long_input_name, TensorProto.FLOAT, [2, 3])
    output_tensor = helper.make_tensor_value_info(long_output_name, TensorProto.FLOAT, [2, 3])

    # Node with long name
    nodes = [
        helper.make_node('Relu', [long_input_name], [long_output_name], name=long_node_name),
    ]

    # Create the graph
    graph = helper.make_graph(
        nodes,
        'very_long_names_model',
        [input_tensor],
        [output_tensor],
    )

    # Create the model
    model = helper.make_model(graph, producer_name="onnx-ir-test", opset_imports=[helper.make_opsetid("", 16)])

    # Check the model
    onnx.checker.check_model(model)

    return model


def main():
    """Generate and save the ONNX model."""
    model = create_very_long_names_model()

    # Save the model
    output_path = '../fixtures/very_long_names.onnx'
    onnx.save(model, output_path)
    print(f"Model saved to {output_path}")

    print(f"\nModel info:")
    print(f"  Node name length: {len(model.graph.node[0].name)} chars")
    print(f"  Input name length: {len(model.graph.input[0].name)} chars")
    print(f"  Output name length: {len(model.graph.output[0].name)} chars")
    print(f"  Tests string handling for 100+ character names")


if __name__ == '__main__':
    main()
