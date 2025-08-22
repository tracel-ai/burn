#!/usr/bin/env python3

"""
Script to generate an ONNX model that uses Shape types as runtime slice parameters.
This demonstrates slicing a tensor using shape values from another tensor.
"""

import torch
import torch.nn as nn
import numpy as np
from onnx import helper, TensorProto, save
import onnx

def build_manual_graph():
    """Build ONNX graph manually using ONNX helper functions"""
    
    # Define the input tensor to be sliced
    input_tensor = helper.make_tensor_value_info(
        'input', TensorProto.FLOAT, [10, 8, 6]
    )
    
    # Define a shape tensor that will provide slice parameters
    shape_tensor = helper.make_tensor_value_info(
        'shape_input', TensorProto.FLOAT, [3, 4]  # Will use its shape [3, 4] as slice params
    )
    
    # Define the output tensor (result of slicing)
    output_tensor = helper.make_tensor_value_info(
        'output', TensorProto.FLOAT, [3, 4, 6]  # Slicing [10,8,6] from [0:3, 0:4, :]
    )
    
    # Create constant for axes specification
    axes_tensor = helper.make_tensor(
        name='axes', 
        data_type=TensorProto.INT64,
        dims=[2],
        vals=[0, 1]  # Slice on axes 0 and 1
    )
    
    # Create nodes
    
    # 1. Shape node: shape_input -> shape
    shape_node = helper.make_node(
        'Shape',
        inputs=['shape_input'],
        outputs=['shape'],
        name='shape'
    )
    
    # 2. Constant node for starts (zeros)
    starts_node = helper.make_node(
        'Constant',
        inputs=[],  # Constant node has no inputs
        outputs=['starts'],
        name='starts',
        value=helper.make_tensor(
            name='starts_value',
            data_type=TensorProto.INT64,
            dims=[2],
            vals=[0, 0]
        )
    )
    
    # 3. Slice node: input, starts, shape (as ends), axes -> output
    # Using the shape [3, 4] directly as the ends parameter
    slice_node = helper.make_node(
        'Slice',
        inputs=['input', 'starts', 'shape', 'axes'],
        outputs=['output'],
        name='slice'
    )
    
    # Create the graph
    graph = helper.make_graph(
        nodes=[shape_node, starts_node, slice_node],
        name='SliceShapeRuntimeGraph',
        inputs=[input_tensor, shape_tensor],
        outputs=[output_tensor],
        initializer=[axes_tensor]
    )
    
    # Create the model
    model = helper.make_model(
        graph, 
        producer_name='slice_shape_runtime',
        opset_imports=[helper.make_operatorsetid('', 16)]
    )
    
    return model

def main():
    print("Generating slice_shape_runtime.onnx")
    
    # Build the model
    model = build_manual_graph()
    
    # Save the model
    save(model, 'slice_shape_runtime.onnx')
    print("Model saved to slice_shape_runtime.onnx")
    
    # Test with ReferenceEvaluator to get actual results
    print("\nTesting with ReferenceEvaluator:")
    try:
        from onnx.reference import ReferenceEvaluator
        
        input_data = np.random.randn(10, 8, 6).astype(np.float32)
        shape_input_data = np.random.randn(3, 4).astype(np.float32)
        
        print(f"Input shape: {input_data.shape}")
        print(f"Shape input shape: {shape_input_data.shape}")
        
        # Run with ReferenceEvaluator to get actual output
        session = ReferenceEvaluator(model, verbose=0)
        actual_output, = session.run(None, {"input": input_data, "shape_input": shape_input_data})
        
        print(f"Actual output shape: {actual_output.shape}")
        print(f"Actual output dtype: {actual_output.dtype}")
        
        # Verify the slicing operation
        shape = np.array(shape_input_data.shape)  # [3, 4]
        print(f"Extracted shape (used as slice ends): {shape}")
        print(f"Slice operation: input[0:{shape[0]}, 0:{shape[1]}, :] -> output[{actual_output.shape}]")
        
        # Manual verification
        expected_output = input_data[0:shape[0], 0:shape[1], :]
        print(f"Manual slice result shape: {expected_output.shape}")
        print(f"Outputs match: {np.allclose(actual_output, expected_output)}")
        
    except ImportError:
        print("onnx.reference not available, skipping ReferenceEvaluator test")
        # Fallback to simulation
        input_data = np.random.randn(10, 8, 6).astype(np.float32)
        shape_input_data = np.random.randn(3, 4).astype(np.float32)
        
        print(f"Input shape: {input_data.shape}")
        print(f"Shape input shape: {shape_input_data.shape}")
        
        shape = np.array(shape_input_data.shape)  # [3, 4]
        print(f"Extracted shape: {shape}")
        print(f"Slice on axes [0, 1]: starts=[0, 0], ends={shape}")
        print(f"Expected output shape: [3, 4, 6]")

if __name__ == "__main__":
    main()