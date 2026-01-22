#!/usr/bin/env python3

# used to generate model: unsqueeze_scalar_axes.onnx
# This test verifies that Unsqueeze handles scalar axes input (not just 1D tensor).
# Some ONNX models provide the axes as a scalar constant instead of a 1D tensor.

import numpy as np
import onnx
from onnx import helper, TensorProto
from onnx.reference import ReferenceEvaluator


def create_unsqueeze_scalar_axes_model():
    """
    Creates an ONNX model where Unsqueeze's axes input is a scalar constant.
    This tests the fix for scalar axes handling in extract_config.
    """

    # Input: 2D tensor
    input_tensor = helper.make_tensor_value_info(
        'input',
        TensorProto.FLOAT,
        [3, 4]
    )

    # Output: 3D tensor (after unsqueeze at axis 0)
    output_tensor = helper.make_tensor_value_info(
        'output',
        TensorProto.FLOAT,
        [1, 3, 4]
    )

    # Create scalar axes constant (not 1D tensor)
    # This is the key - dims=[] makes it a scalar
    axes_const = helper.make_node(
        'Constant',
        inputs=[],
        outputs=['axes'],
        value=helper.make_tensor(
            name='axes_value',
            data_type=TensorProto.INT64,
            dims=[],  # Scalar - this triggers the bug we fixed
            vals=[0]
        )
    )

    # Unsqueeze operation with scalar axes
    unsqueeze_node = helper.make_node(
        'Unsqueeze',
        inputs=['input', 'axes'],
        outputs=['output']
    )

    # Create the graph
    graph = helper.make_graph(
        [axes_const, unsqueeze_node],
        'unsqueeze_scalar_axes',
        [input_tensor],
        [output_tensor]
    )

    # Create the model
    model = helper.make_model(
        graph,
        producer_name='unsqueeze_scalar_axes_test',
        opset_imports=[helper.make_operatorsetid("", 16)]
    )
    model.ir_version = 8

    return model


def main():
    model = create_unsqueeze_scalar_axes_model()
    onnx.save(model, "unsqueeze_scalar_axes.onnx")
    print("Finished exporting model to unsqueeze_scalar_axes.onnx")

    # Verify the model
    onnx.checker.check_model(model)

    # Test with ReferenceEvaluator
    try:
        session = ReferenceEvaluator(model, verbose=0)

        # Test input: 2D tensor [3, 4]
        test_input = np.ones((3, 4), dtype=np.float32)
        print(f"\nTest input shape: {test_input.shape}")

        # Run inference
        output, = session.run(None, {"input": test_input})

        print(f"Test output shape: {output.shape}")

        # Verify the result
        expected_shape = (1, 3, 4)
        assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"
        print(f"Test passed: {test_input.shape} unsqueezed to {output.shape}")

    except Exception as e:
        print(f"\nError with ReferenceEvaluator: {e}")


if __name__ == "__main__":
    main()
