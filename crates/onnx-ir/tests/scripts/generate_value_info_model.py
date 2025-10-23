#!/usr/bin/env -S uv run
# /// script
# dependencies = [
#   "onnx>=1.15.0",
#   "numpy>=1.24.0",
# ]
# ///

"""
Generate ONNX model with value_info for intermediate values.

This test validates that the ONNX-IR pipeline correctly uses value_info
to initialize node output types, rather than relying on default rank-0 types.

The model has:
- 2 inputs (rank 2 and rank 3)
- Reshape nodes that change ranks
- Transpose nodes that require tensor inputs (not scalars)
- value_info entries for all intermediate values with correct ranks
"""

import onnx
from onnx import helper, TensorProto
import numpy as np


def create_model():
    # Input 1: rank 2 tensor [batch, features]
    input1 = helper.make_tensor_value_info('input1', TensorProto.FLOAT, ['batch', 784])

    # Input 2: rank 3 tensor for reshape [batch, height, width]
    input2 = helper.make_tensor_value_info('input2', TensorProto.FLOAT, ['batch', 28, 28])

    # Node 1: Reshape input1 from [batch, 784] to [batch, 28, 28]
    # Shape is dynamic (comes from a Concat node)
    shape1_values = np.array([0, 28, 28], dtype=np.int64)  # 0 means copy from input
    shape1 = helper.make_tensor('shape1', TensorProto.INT64, [3], shape1_values)

    reshape1 = helper.make_node(
        'Reshape',
        inputs=['input1', 'shape1'],
        outputs=['reshape1_out'],
        name='reshape1'
    )

    # Node 2: Transpose reshape1_out from [batch, 28, 28] to [batch, 28, 28] (swap last two dims)
    transpose1 = helper.make_node(
        'Transpose',
        inputs=['reshape1_out'],
        outputs=['transpose1_out'],
        perm=[0, 2, 1],
        name='transpose1'
    )

    # Node 3: Add the transposed tensor with input2
    add1 = helper.make_node(
        'Add',
        inputs=['transpose1_out', 'input2'],
        outputs=['add1_out'],
        name='add1'
    )

    # Node 4: Reshape back to [batch, 784]
    shape2_values = np.array([0, 784], dtype=np.int64)
    shape2 = helper.make_tensor('shape2', TensorProto.INT64, [2], shape2_values)

    reshape2 = helper.make_node(
        'Reshape',
        inputs=['add1_out', 'shape2'],
        outputs=['output1'],
        name='reshape2'
    )

    # Output
    output1 = helper.make_tensor_value_info('output1', TensorProto.FLOAT, ['batch', 784])

    # CRITICAL: value_info for intermediate values
    # This is what we're testing - that ONNX-IR uses these to initialize node outputs
    # instead of using default rank-0 types
    value_info = [
        # reshape1_out should be rank 3, NOT rank 0!
        helper.make_tensor_value_info('reshape1_out', TensorProto.FLOAT, ['batch', 28, 28]),
        # transpose1_out should be rank 3
        helper.make_tensor_value_info('transpose1_out', TensorProto.FLOAT, ['batch', 28, 28]),
        # add1_out should be rank 3
        helper.make_tensor_value_info('add1_out', TensorProto.FLOAT, ['batch', 28, 28]),
    ]

    # Create the graph
    graph = helper.make_graph(
        nodes=[reshape1, transpose1, add1, reshape2],
        name='value_info_test',
        inputs=[input1, input2],
        outputs=[output1],
        initializer=[shape1, shape2],
        value_info=value_info  # THIS IS THE KEY PART WE'RE TESTING
    )

    # Create the model
    model = helper.make_model(graph, producer_name='value_info_test')
    model.opset_import[0].version = 16

    # Validate and save
    onnx.checker.check_model(model)

    output_path = '../fixtures/value_info.onnx'
    onnx.save(model, output_path)
    print(f"✓ Saved model to {output_path}")
    print(f"  - 2 inputs: input1 [batch, 784], input2 [batch, 28, 28]")
    print(f"  - 4 nodes: Reshape → Transpose → Add → Reshape")
    print(f"  - 3 value_info entries for intermediate values")
    print(f"  - Tests that Transpose receives Tensor (rank 3), not Scalar (rank 0)")


if __name__ == '__main__':
    create_model()
