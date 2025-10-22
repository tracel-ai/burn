#!/usr/bin/env -S uv run
# /// script
# dependencies = [
#   "onnx>=1.15.0",
#   "numpy>=1.24.0",
# ]
# ///

"""
Generate ONNX model testing edge cases.

Tests:
- Scalar initializers (empty shape and shape=[1])
- Single-element tensors
- Different scalar representations
- Edge cases in constant handling
"""

import onnx
from onnx import helper, TensorProto
import numpy as np


def create_edge_cases_model():
    """Create model with various edge cases."""

    # Input
    input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, [2, 3])

    # Outputs
    output1 = helper.make_tensor_value_info('output1', TensorProto.FLOAT, [2, 3])
    output2 = helper.make_tensor_value_info('output2', TensorProto.FLOAT, [2, 3])
    output3 = helper.make_tensor_value_info('output3', TensorProto.FLOAT, [2, 3])

    # Edge case 1: Scalar with empty shape []
    scalar_empty_shape = helper.make_tensor(
        name='scalar_empty',
        data_type=TensorProto.FLOAT,
        dims=[],  # Empty dims = scalar
        vals=np.array([5.0], dtype=np.float32).tobytes(),
        raw=True
    )

    # Edge case 2: Scalar with shape [1]
    scalar_shape_1 = helper.make_tensor(
        name='scalar_one',
        data_type=TensorProto.FLOAT,
        dims=[1],  # Single element
        vals=np.array([3.0], dtype=np.float32).tobytes(),
        raw=True
    )

    # Edge case 3: Single-element tensor with shape [1, 1]
    single_elem_2d = helper.make_tensor(
        name='single_2d',
        data_type=TensorProto.FLOAT,
        dims=[1, 1],
        vals=np.array([[2.0]], dtype=np.float32).flatten().tobytes(),
        raw=True
    )

    # Create nodes testing edge cases
    nodes = [
        # Use scalar with empty shape
        helper.make_node('Add', ['input', 'scalar_empty'], ['output1'], name='add1'),

        # Use scalar with shape [1]
        helper.make_node('Mul', ['input', 'scalar_one'], ['output2'], name='mul1'),

        # Use single-element 2D tensor
        helper.make_node('Add', ['input', 'single_2d'], ['output3'], name='add2'),
    ]

    # Create the graph
    graph = helper.make_graph(
        nodes,
        'edge_cases_model',
        [input_tensor],
        [output1, output2, output3],
        initializer=[scalar_empty_shape, scalar_shape_1, single_elem_2d]
    )

    # Create the model
    model = helper.make_model(graph, producer_name='onnx-ir-test')
    model.opset_import[0].version = 16

    # Check the model
    onnx.checker.check_model(model)

    return model


def main():
    """Generate and save the ONNX model."""
    model = create_edge_cases_model()

    # Save the model
    output_path = '../fixtures/edge_cases.onnx'
    onnx.save(model, output_path)
    print(f"Model saved to {output_path}")

    # Print model info
    print(f"\nModel info:")
    print(f"  Opset version: {model.opset_import[0].version}")
    print(f"  Inputs: {[inp.name for inp in model.graph.input]}")
    print(f"  Outputs: {[out.name for out in model.graph.output]}")
    print(f"  Initializers:")
    for init in model.graph.initializer:
        print(f"    - {init.name}: shape={list(init.dims)}, dtype={init.data_type}")
    print(f"  Nodes: {len(model.graph.node)}")
    for node in model.graph.node:
        print(f"    - {node.op_type} ({node.name}): {list(node.input)} → {list(node.output)}")


if __name__ == '__main__':
    main()
