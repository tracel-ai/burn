#!/usr/bin/env -S uv run
# /// script
# dependencies = [
#   "onnx>=1.15.0",
#   "numpy>=1.24.0",
# ]
# ///

"""
Generate ONNX model with passthrough (input directly to output via Identity).

Tests:
- Graph where input flows directly to output
- Identity elimination edge case
- Minimal graph structure
"""

import onnx
from onnx import helper, TensorProto


def create_passthrough_model():
    """Create model where input passes through to output."""

    # Input and output with same shape
    input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 4])
    output_tensor = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 4])

    # Single Identity node
    nodes = [
        helper.make_node('Identity', ['input'], ['output'], name='passthrough'),
    ]

    # Create the graph
    graph = helper.make_graph(
        nodes,
        'passthrough_model',
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
    model = create_passthrough_model()

    # Save the model
    output_path = '../fixtures/passthrough.onnx'
    onnx.save(model, output_path)
    print(f"Model saved to {output_path}")

    print(f"\nModel info:")
    print(f"  Single Identity node connecting input to output")
    print(f"  Should be optimized away in Phase 4")


if __name__ == '__main__':
    main()
