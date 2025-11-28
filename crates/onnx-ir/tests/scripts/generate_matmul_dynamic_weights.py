#!/usr/bin/env -S uv run
# /// script
# dependencies = [
#   "onnx>=1.15.0",
#   "numpy>=1.24.0",
# ]
# ///

"""
Generate ONNX model with MatMul using dynamic (runtime) weights.

Tests:
- MatMul that cannot be coalesced to Linear (weights not constant)
- Edge case #18: MatMul with dynamic weights (no coalesce)
"""

import onnx
from onnx import helper, TensorProto


def create_matmul_dynamic_weights_model():
    """Create model where MatMul uses runtime weights (not constants)."""

    # Both inputs are runtime (not constants)
    input_data = helper.make_tensor_value_info('input_data', TensorProto.FLOAT, [2, 3])
    input_weights = helper.make_tensor_value_info('input_weights', TensorProto.FLOAT, [3, 4])

    # Output
    output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [2, 4])

    # MatMul with both inputs being runtime (not constant)
    # This should NOT be coalesced to Linear because weights are not static
    nodes = [
        helper.make_node('MatMul', ['input_data', 'input_weights'], ['output'], name='matmul_dynamic'),
    ]

    # Create the graph (no initializers - all inputs are runtime)
    graph = helper.make_graph(
        nodes,
        'matmul_dynamic_weights_model',
        [input_data, input_weights],
        [output],
    )

    # Create the model
    model = helper.make_model(graph, producer_name="onnx-ir-test", opset_imports=[helper.make_opsetid("", 16)])

    # Check the model
    onnx.checker.check_model(model)

    return model


def main():
    """Generate and save the ONNX model."""
    model = create_matmul_dynamic_weights_model()

    # Save the model
    output_path = '../fixtures/matmul_dynamic_weights.onnx'
    onnx.save(model, output_path)
    print(f"Model saved to {output_path}")

    print(f"\nModel info:")
    print(f"  MatMul with runtime (dynamic) weights")
    print(f"  Should NOT coalesce to Linear (no constant weights)")


if __name__ == '__main__':
    main()
