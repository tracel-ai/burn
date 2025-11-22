#!/usr/bin/env -S uv run
# /// script
# dependencies = [
#   "onnx>=1.15.0",
#   "numpy>=1.24.0",
# ]
# ///

"""
Generate ONNX model with large constants (MB-sized).

Tests:
- Handling of large constant tensors
- Edge case #32: Large constants (MB-sized)
"""

import onnx
from onnx import helper, TensorProto
import numpy as np


def create_large_constants_model():
    """Create model with large constant tensors."""

    # Input (small)
    input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 512])

    # Output
    output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 512])

    # Large weight matrix: 512x512 floats = 1MB
    large_weight = helper.make_tensor(
        name='large_weight',
        data_type=TensorProto.FLOAT,
        dims=[512, 512],
        vals=np.random.randn(512, 512).astype(np.float32).tobytes(),
        raw=True
    )

    # Another large constant: 512 floats
    large_bias = helper.make_tensor(
        name='large_bias',
        data_type=TensorProto.FLOAT,
        dims=[512],
        vals=np.ones(512, dtype=np.float32).tobytes(),
        raw=True
    )

    # Operations using large constants
    nodes = [
        helper.make_node('MatMul', ['input', 'large_weight'], ['temp1'], name='matmul_large'),
        helper.make_node('Add', ['temp1', 'large_bias'], ['output'], name='add_large'),
    ]

    # Create the graph
    graph = helper.make_graph(
        nodes,
        'large_constants_model',
        [input_tensor],
        [output],
        initializer=[large_weight, large_bias]
    )

    # Create the model
    model = helper.make_model(graph, producer_name="onnx-ir-test", opset_imports=[helper.make_opsetid("", 16)])

    # Check the model
    onnx.checker.check_model(model)

    return model


def main():
    """Generate and save the ONNX model."""
    model = create_large_constants_model()

    # Save the model
    output_path = '../fixtures/large_constants.onnx'
    onnx.save(model, output_path)
    print(f"Model saved to {output_path}")

    import os
    file_size = os.path.getsize(output_path)
    print(f"\nModel info:")
    print(f"  File size: {file_size / 1024 / 1024:.2f} MB")
    print(f"  Large weight: 512x512 floats = ~1 MB")
    print(f"  Large bias: 512 floats = ~2 KB")
    print(f"  Tests handling of MB-sized constant tensors")


if __name__ == '__main__':
    main()
