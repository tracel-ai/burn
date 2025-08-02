#!/usr/bin/env python3

"""
Script to generate an ONNX model that uses multi-dimensional Shape types as slice parameters.
This demonstrates slicing a 4D tensor using shape values that provide multiple dimension limits.
"""

import numpy as np
from onnx import helper, TensorProto, save

def build_manual_graph():
    """Build ONNX graph manually using ONNX helper functions"""
    
    # Define the input tensor to be sliced (4D)
    input_tensor = helper.make_tensor_value_info(
        'input', TensorProto.FLOAT, [8, 6, 10, 12]
    )
    
    # Define shape tensors that will provide slice start/end parameters
    start_shape_tensor = helper.make_tensor_value_info(
        'start_shape_input', TensorProto.FLOAT, [1, 2, 3]  # Shape [1, 2, 3] as start indices
    )
    
    end_shape_tensor = helper.make_tensor_value_info(
        'end_shape_input', TensorProto.FLOAT, [5, 4, 7]  # Shape [5, 4, 7] as end indices
    )
    
    # Define the output tensor (result of slicing)
    # Slicing [8,6,10,12] from [1:5, 2:4, 3:7, :] = [4, 2, 4, 12]
    output_tensor = helper.make_tensor_value_info(
        'output', TensorProto.FLOAT, [4, 2, 4, 12]
    )
    
    # Create constant for axes specification
    axes_tensor = helper.make_tensor(
        name='axes', 
        data_type=TensorProto.INT64,
        dims=[3],
        vals=[0, 1, 2]  # Slice on first 3 axes
    )
    
    # Create nodes
    
    # 1. Shape node for start indices: start_shape_input -> start_shape
    start_shape_node = helper.make_node(
        'Shape',
        inputs=['start_shape_input'],
        outputs=['start_shape'],
        name='start_shape'
    )
    
    # 2. Shape node for end indices: end_shape_input -> end_shape
    end_shape_node = helper.make_node(
        'Shape',
        inputs=['end_shape_input'],
        outputs=['end_shape'],
        name='end_shape'
    )
    
    # 3. Slice node: input, start_shape, end_shape, axes -> output
    slice_node = helper.make_node(
        'Slice',
        inputs=['input', 'start_shape', 'end_shape', 'axes'],
        outputs=['output'],
        name='slice'
    )
    
    # Create the graph
    graph = helper.make_graph(
        nodes=[start_shape_node, end_shape_node, slice_node],
        name='SliceShapeMultiGraph',
        inputs=[input_tensor, start_shape_tensor, end_shape_tensor],
        outputs=[output_tensor],
        initializer=[axes_tensor]
    )
    
    # Create the model
    model = helper.make_model(graph, producer_name='slice_shape_multi')
    model.opset_import[0].version = 16
    
    return model

def main():
    print("Generating slice_shape_multi.onnx")
    
    # Build the model
    model = build_manual_graph()
    
    # Save the model
    save(model, 'slice_shape_multi.onnx')
    print("Model saved to slice_shape_multi.onnx")
    
    # Test with dummy data
    print("\nTesting with dummy data:")
    input_data = np.random.randn(8, 6, 10, 12).astype(np.float32)
    start_shape_data = np.random.randn(1, 2, 3).astype(np.float32)
    end_shape_data = np.random.randn(5, 4, 7).astype(np.float32)
    
    print(f"Input shape: {input_data.shape}")
    print(f"Start shape input: {start_shape_data.shape}")
    print(f"End shape input: {end_shape_data.shape}")
    
    # Simulate the operations
    start_shape = np.array(start_shape_data.shape)  # [1, 2, 3]
    end_shape = np.array(end_shape_data.shape)      # [5, 4, 7]
    print(f"Start indices from shape: {start_shape}")
    print(f"End indices from shape: {end_shape}")
    
    # The slice uses shapes as start/end indices for axes [0, 1, 2]
    print(f"Slice on axes [0, 1, 2]: starts={start_shape}, ends={end_shape}")
    print(f"Expected output shape: [4, 2, 4, 12]")

if __name__ == "__main__":
    main()