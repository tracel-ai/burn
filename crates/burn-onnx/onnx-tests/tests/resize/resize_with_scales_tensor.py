#!/usr/bin/env python3

# used to generate model: onnx-tests/tests/resize/resize_with_scales_tensor.onnx
# This tests resize with a runtime tensor input for scales (not a constant)
# This is the counterpart to resize_with_sizes_tensor.py but for scales instead of sizes

import onnx
from onnx import helper, TensorProto
import numpy as np

def main() -> None:
    # Create input tensors
    input_tensor = helper.make_tensor_value_info("input_tensor", TensorProto.FLOAT, [1, 3, 4, 4])

    # Create scales as an input tensor (not a constant) - this will be provided at runtime
    # Scales format: [scale_n, scale_c, scale_h, scale_w]
    scales_input = helper.make_tensor_value_info("scales_input", TensorProto.FLOAT, [4])

    # Create Resize node that takes the scales as a runtime tensor input
    # inputs: [X, roi, scales, sizes] - we provide scales at index 2, leave sizes empty
    resize_node = helper.make_node(
        "Resize",
        name="resize_node",
        inputs=["input_tensor", "", "scales_input", ""],  # scales_input is a runtime tensor
        outputs=["output"],
        mode="nearest",  # Use nearest for simplicity
    )

    graph_def = helper.make_graph(
        nodes=[resize_node],
        name="ResizeWithTensorScalesGraph",
        inputs=[input_tensor, scales_input],  # Both are inputs to the graph
        outputs=[
            helper.make_tensor_value_info("output", TensorProto.FLOAT, None)  # Dynamic output shape
        ],
        initializer=[],  # No initializers - scales_input is a runtime input
    )

    model_def = helper.make_model(
        graph_def,
        producer_name="resize_with_scales_tensor",
        opset_imports=[helper.make_operatorsetid("", 16)]
    )

    onnx.save(model_def, "resize_with_scales_tensor.onnx")

    # Verify with onnx.reference.ReferenceEvaluator
    try:
        from onnx.reference import ReferenceEvaluator

        # Test input [1, 3, 4, 4]
        test_input = np.array([[
            [
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0, 12.0],
                [13.0, 14.0, 15.0, 16.0],
            ],
            [
                [17.0, 18.0, 19.0, 20.0],
                [21.0, 22.0, 23.0, 24.0],
                [25.0, 26.0, 27.0, 28.0],
                [29.0, 30.0, 31.0, 32.0],
            ],
            [
                [33.0, 34.0, 35.0, 36.0],
                [37.0, 38.0, 39.0, 40.0],
                [41.0, 42.0, 43.0, 44.0],
                [45.0, 46.0, 47.0, 48.0],
            ],
        ]], dtype=np.float32)

        # Scales tensor to double spatial dimensions: [1, 3, 4, 4] -> [1, 3, 8, 8]
        # Format: [scale_n, scale_c, scale_h, scale_w]
        scales_tensor = np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32)

        print(f"Test input shape: {test_input.shape}")
        print(f"Test scales tensor: {scales_tensor}")

        # Run inference with ONNX model
        sess = ReferenceEvaluator(model_def)
        result = sess.run(None, {"input_tensor": test_input, "scales_input": scales_tensor})

        print(f"ONNX model output shape: {result[0].shape}")
        print(f"ONNX model output sum: {result[0].sum()}")
        print(f"ONNX model output dtype: {result[0].dtype}")

        # Print corner values for verification
        print(f"First channel corners: [{result[0][0,0,0,0]}, {result[0][0,0,0,-1]}, {result[0][0,0,-1,0]}, {result[0][0,0,-1,-1]}]")

    except ImportError:
        print("onnx.reference not available, skipping ONNX model verification")


if __name__ == "__main__":
    main()
