#!/usr/bin/env python3

# used to generate model: onnx-tests/tests/expand/expand_max_semantics.onnx
#
# Tests ONNX Expand's max-semantics behavior:
# When shape_dim=1 but input_dim>1, ONNX keeps the input_dim (not replaces with 1).
# This is different from simple broadcasting.

import numpy as np
import onnx
from onnx import helper, TensorProto
from onnx.reference import ReferenceEvaluator

# ONNX opset version to use for model generation
OPSET_VERSION = 16


def main() -> None:
    # Define the shape tensor as a constant node
    # Shape [1, 1] with input [2, 3] should output [2, 3] (max semantics)
    shape_value = [1, 1]
    shape_tensor = helper.make_tensor(
        name='shape',
        data_type=TensorProto.INT64,
        dims=[len(shape_value)],
        vals=shape_value,
    )

    shape_node = helper.make_node(
        'Constant',
        name='shape_constant',
        inputs=[],
        outputs=['shape'],
        value=shape_tensor,
    )

    # Define the Expand node
    expand_node = helper.make_node(
        'Expand',
        name='/Expand',
        inputs=['input_tensor', 'shape'],
        outputs=['output']
    )

    # Create the graph
    # Input: [2, 3], Shape: [1, 1], Output: [2, 3] (max semantics)
    graph_def = helper.make_graph(
        nodes=[shape_node, expand_node],
        name='ExpandMaxSemanticsGraph',
        inputs=[
            helper.make_tensor_value_info('input_tensor', TensorProto.FLOAT, [2, 3]),
        ],
        outputs=[
            helper.make_tensor_value_info('output', TensorProto.FLOAT, [2, 3])
        ],
    )

    # Create the model
    model_def = helper.make_model(
        graph_def,
        producer_name='expand_max_semantics',
        opset_imports=[helper.make_operatorsetid("", OPSET_VERSION)]
    )

    # Ensure valid ONNX:
    onnx.checker.check_model(model_def)

    # Save the model to a file
    onnx_name = 'expand_max_semantics.onnx'
    onnx.save(model_def, onnx_name)
    print(f"Finished exporting model to {onnx_name}")

    # Test the model with sample data
    test_input = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    print(f"\nTest input shape: {test_input.shape}")
    print(f"Test input:\n{test_input}")

    # Run the model using ReferenceEvaluator
    session = ReferenceEvaluator(onnx_name, verbose=0)
    outputs = session.run(None, {"input_tensor": test_input})

    output = outputs[0]
    print(f"\nTest output shape: {output.shape}")
    print(f"Test output:\n{output}")

    # Verify max-semantics: output should be [2, 3], same as input
    assert output.shape == (2, 3), f"Expected shape (2, 3), got {output.shape}"
    assert np.allclose(output, test_input), "Output should equal input (no broadcasting)"
    print("\nMax-semantics verification passed!")


if __name__ == '__main__':
    main()
