#!/usr/bin/env python3

import torch
import torch.nn as nn
import numpy as np
from onnx import helper, TensorProto, save, ValueInfoProto

def build_manual_graph():
    """Build ONNX graph manually using ONNX helper functions"""
    
    # Define the input tensor
    input_tensor = helper.make_tensor_value_info(
        'input', TensorProto.FLOAT, [2, 4, 6, 8]
    )
    
    # Define the output tensor (sliced result)
    output_tensor = helper.make_tensor_value_info(
        'output', TensorProto.FLOAT, [2, 4, 6, 8]  # Will slice dim 1 from 0 to 4 (full dimension)
    )
    
    # Create constant tensors for Slice operation
    # starts: [0] - slice from beginning of dim 1
    starts_tensor = helper.make_tensor(
        name='starts',
        data_type=TensorProto.INT64,
        dims=[1],
        vals=[0]
    )
    
    # axes: [1] - slice along axis 1
    axes_tensor = helper.make_tensor(
        name='axes', 
        data_type=TensorProto.INT64,
        dims=[1],
        vals=[1]
    )
    
    # steps: [1] - step size 1
    steps_tensor = helper.make_tensor(
        name='steps',
        data_type=TensorProto.INT64, 
        dims=[1],
        vals=[1]
    )
    
    # Gather indices: [1] - get shape dimension 1 
    gather_indices = helper.make_tensor(
        name='gather_indices',
        data_type=TensorProto.INT64,
        dims=[],  # Scalar
        vals=[1]
    )
    
    # Create the nodes
    
    # 1. Shape node: input -> shape
    shape_node = helper.make_node(
        'Shape',
        inputs=['input'],
        outputs=['shape'],
        name='shape'
    )
    
    # 2. Gather node: shape, gather_indices -> gathered_dim (scalar)
    gather_node = helper.make_node(
        'Gather',
        inputs=['shape', 'gather_indices'],
        outputs=['gathered_dim'],
        name='gather',
        axis=0
    )
    
    # 3. Slice node: input, starts, gathered_dim (as ends), axes, steps -> output
    # Using gathered_dim directly as ends (which should be 4 from the shape)
    slice_node = helper.make_node(
        'Slice',
        inputs=['input', 'starts', 'gathered_dim', 'axes', 'steps'],
        outputs=['output'],
        name='slice'
    )
    
    # Create the graph
    graph = helper.make_graph(
        nodes=[shape_node, gather_node, slice_node],
        name='SliceShapeGatherGraph',
        inputs=[input_tensor],
        outputs=[output_tensor],
        initializer=[starts_tensor, axes_tensor, steps_tensor, gather_indices]
    )
    
    # Create the model
    model = helper.make_model(graph, producer_name='slice_shape_gather')
    
    return model

def main():
    print("Generating slice_shape_gather.onnx")
    
    # Build the model
    model = build_manual_graph()
    
    # Save the model
    save(model, 'slice_shape_gather.onnx')
    print("Model saved to slice_shape_gather.onnx")
    
    # Test with dummy data
    print("\nTesting with dummy data:")
    input_data = np.random.randn(2, 4, 6, 8).astype(np.float32)
    print(f"Input shape: {input_data.shape}")
    
    # Simulate the operations
    shape = np.array(input_data.shape)
    print(f"Shape: {shape}")
    
    gathered_dim = shape[1]  # Gather axis=0, indices=[1]
    print(f"Gathered dimension: {gathered_dim}")
    
    # The slice uses gathered_dim as ends, slicing axis 1 from 0 to gathered_dim
    # But we want to slice to 3, not 4, so let's update the expected output
    print(f"Slice on axis 1: starts=[0], ends=[{gathered_dim}]")
    print(f"Note: Output expects shape [2, 3, 6, 8], so actual slice would be 0:3")

if __name__ == "__main__":
    main()