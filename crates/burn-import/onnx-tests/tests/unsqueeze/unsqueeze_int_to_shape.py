#!/usr/bin/env python3

# used to generate model: unsqueeze_int_to_shape.onnx

import numpy as np
import onnx
from onnx import helper, TensorProto
import onnxruntime as ort


def create_unsqueeze_int_to_shape_model():
    """
    Creates an ONNX model that unsqueezes an int64 scalar to a shape.
    This demonstrates the reverse of squeeze(Shape[1]) -> Scalar.
    """
    
    # Input: int64 scalar
    input_scalar = helper.make_tensor_value_info(
        'scalar_input', 
        TensorProto.INT64, 
        []  # scalar has no dimensions
    )
    
    # Output: Shape[1] (1D array with one element)
    output_shape = helper.make_tensor_value_info(
        'shape_output', 
        TensorProto.INT64, 
        [1]  # 1D tensor with size 1
    )
    
    # Create axes constant for unsqueeze (add dimension at axis 0)
    axes_const = helper.make_node(
        'Constant',
        inputs=[],
        outputs=['axes'],
        value=helper.make_tensor(
            name='axes_value',
            data_type=TensorProto.INT64,
            dims=[1],
            vals=[0]
        )
    )
    
    # Unsqueeze operation: scalar -> shape[1]
    unsqueeze_node = helper.make_node(
        'Unsqueeze',
        inputs=['scalar_input', 'axes'],
        outputs=['shape_output']
    )
    
    # Create the graph
    graph = helper.make_graph(
        [axes_const, unsqueeze_node],
        'unsqueeze_int_to_shape',
        [input_scalar],
        [output_shape]
    )
    
    # Create the model
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 16)])
    model.ir_version = 8  # Use IR version 8 for compatibility
    model.producer_name = 'unsqueeze_int_to_shape_test'
    model.producer_version = '1.0'
    
    return model


def create_squeeze_unsqueeze_roundtrip_model():
    """
    Creates an ONNX model that demonstrates the squeeze/unsqueeze roundtrip:
    Shape[1] -> squeeze -> Scalar -> unsqueeze -> Shape[1]
    """
    
    # Input: Shape[1] (1D array with one element)
    input_shape = helper.make_tensor_value_info(
        'shape_input', 
        TensorProto.INT64, 
        [1]
    )
    
    # Output: Shape[1] after roundtrip
    output_shape = helper.make_tensor_value_info(
        'shape_output', 
        TensorProto.INT64, 
        [1]
    )
    
    # Squeeze axes constant
    squeeze_axes_const = helper.make_node(
        'Constant',
        inputs=[],
        outputs=['squeeze_axes'],
        value=helper.make_tensor(
            name='squeeze_axes_value',
            data_type=TensorProto.INT64,
            dims=[1],
            vals=[0]
        )
    )
    
    # Squeeze operation: Shape[1] -> Scalar
    squeeze_node = helper.make_node(
        'Squeeze',
        inputs=['shape_input', 'squeeze_axes'],
        outputs=['scalar_intermediate']
    )
    
    # Unsqueeze axes constant
    unsqueeze_axes_const = helper.make_node(
        'Constant',
        inputs=[],
        outputs=['unsqueeze_axes'],
        value=helper.make_tensor(
            name='unsqueeze_axes_value',
            data_type=TensorProto.INT64,
            dims=[1],
            vals=[0]
        )
    )
    
    # Unsqueeze operation: Scalar -> Shape[1]
    unsqueeze_node = helper.make_node(
        'Unsqueeze',
        inputs=['scalar_intermediate', 'unsqueeze_axes'],
        outputs=['shape_output']
    )
    
    # Create the graph
    graph = helper.make_graph(
        [squeeze_axes_const, squeeze_node, unsqueeze_axes_const, unsqueeze_node],
        'squeeze_unsqueeze_roundtrip',
        [input_shape],
        [output_shape]
    )
    
    # Create the model
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 16)])
    model.ir_version = 8  # Use IR version 8 for compatibility
    model.producer_name = 'squeeze_unsqueeze_roundtrip_test'
    model.producer_version = '1.0'
    
    return model


def main():
    # Create and save the first model (int to shape)
    model1 = create_unsqueeze_int_to_shape_model()
    onnx.save(model1, "unsqueeze_int_to_shape.onnx")
    print("Finished exporting model to unsqueeze_int_to_shape.onnx")
    
    # Verify the model
    onnx.checker.check_model(model1)
    
    # Test with ONNX Runtime
    session = ort.InferenceSession("unsqueeze_int_to_shape.onnx")
    
    # Test input: scalar int64 (0-dimensional array for ONNX runtime)
    test_input = np.array(42, dtype=np.int64)
    print(f"\nTest input data: {test_input}")
    print(f"Test input shape: {test_input.shape}")
    print(f"Test input type: {test_input.dtype}")
    
    # Run inference
    outputs = session.run(None, {"scalar_input": test_input})
    output = outputs[0]
    
    print(f"Test output data: {output}")
    print(f"Test output shape: {output.shape}")
    print(f"Test output type: {output.dtype}")
    
    # Create and save the second model (roundtrip)
    model2 = create_squeeze_unsqueeze_roundtrip_model()
    onnx.save(model2, "squeeze_unsqueeze_roundtrip.onnx")
    print("\nFinished exporting model to squeeze_unsqueeze_roundtrip.onnx")
    
    # Verify the model
    onnx.checker.check_model(model2)
    
    # Test with ONNX Runtime
    session2 = ort.InferenceSession("squeeze_unsqueeze_roundtrip.onnx")
    
    # Test input: 1D array with one element
    test_input2 = np.array([256], dtype=np.int64)
    print(f"\nTest input data: {test_input2}")
    print(f"Test input shape: {test_input2.shape}")
    
    # Run inference
    outputs2 = session2.run(None, {"shape_input": test_input2})
    output2 = outputs2[0]
    
    print(f"Test output data: {output2}")
    print(f"Test output shape: {output2.shape}")
    
    # Verify roundtrip preserves value
    assert np.array_equal(test_input2, output2), "Roundtrip failed to preserve value"
    print("Roundtrip test passed: input == output")


if __name__ == "__main__":
    main()