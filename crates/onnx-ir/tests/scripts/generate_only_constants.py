#!/usr/bin/env -S uv run
# /// script
# dependencies = [
#   "onnx>=1.15.0",
#   "numpy>=1.24.0",
# ]
# ///

"""
Generate ONNX model with only constant nodes (no operations).

Tests:
- Graph with all constant nodes
- Phase 5 unreferenced constant removal edge case
- Graph output directly from initializer
"""

import onnx
from onnx import helper, TensorProto
import numpy as np


def create_only_constants_model():
    """Create model where outputs come directly from initializers."""

    # Graph inputs (will have an input but outputs come from constants)
    input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 3])

    # Graph outputs (come from constants)
    output1 = helper.make_tensor_value_info('output1', TensorProto.FLOAT, [1, 3])
    output2 = helper.make_tensor_value_info('output2', TensorProto.FLOAT, [1, 3])

    # Constants that will be the outputs
    const1 = helper.make_tensor(
        name='const1',
        data_type=TensorProto.FLOAT,
        dims=[1, 3],
        vals=np.array([[1.0, 2.0, 3.0]], dtype=np.float32).flatten().tobytes(),
        raw=True
    )

    const2 = helper.make_tensor(
        name='const2',
        data_type=TensorProto.FLOAT,
        dims=[1, 3],
        vals=np.array([[4.0, 5.0, 6.0]], dtype=np.float32).flatten().tobytes(),
        raw=True
    )

    # Nodes - just Identity to connect constants to outputs
    # (ONNX requires operations, can't directly output initializers)
    nodes = [
        helper.make_node('Identity', ['const1'], ['output1'], name='id1'),
        helper.make_node('Identity', ['const2'], ['output2'], name='id2'),
    ]

    # Create the graph
    graph = helper.make_graph(
        nodes,
        'only_constants_model',
        [input_tensor],
        [output1, output2],
        initializer=[const1, const2]
    )

    # Create the model
    model = helper.make_model(graph, producer_name="onnx-ir-test", opset_imports=[helper.make_opsetid("", 16)])

    # Check the model
    onnx.checker.check_model(model)

    return model


def main():
    """Generate and save the ONNX model."""
    model = create_only_constants_model()

    # Save the model
    output_path = '../fixtures/only_constants.onnx'
    onnx.save(model, output_path)
    print(f"Model saved to {output_path}")

    print(f"\nModel info:")
    print(f"  Nodes: {len(model.graph.node)} (Identity nodes connecting constants)")
    print(f"  Initializers: {len(model.graph.initializer)}")
    print(f"  Outputs come from constants via Identity")


if __name__ == '__main__':
    main()
