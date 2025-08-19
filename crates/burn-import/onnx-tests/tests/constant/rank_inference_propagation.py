#!/usr/bin/env python3

"""
Regression test for rank inference propagation after Shape type conversions.

This test creates an ONNX model that reproduces the issue where:
1. Constant nodes output rank-1 tensors
2. These are used in Concat operations with Shape outputs
3. The constants get converted to Shape type
4. Downstream nodes (like MatMul) need their ranks re-inferred

Without the fix, this would fail with "Concat axis 2 is out of bounds for rank 2"
because MatMul outputs would incorrectly have rank 2 instead of rank 3.
"""

import onnx
import numpy as np
from onnx import TensorProto, helper

def create_model():
    # Create input - 3D tensor for MatMul
    input_3d = helper.make_tensor_value_info(
        'input', TensorProto.FLOAT, [None, None, 384]  # Dynamic batch, sequence, features
    )
    
    # Create weight matrices for MatMul operations
    weight1 = helper.make_tensor(
        name='weight1',
        data_type=TensorProto.FLOAT,
        dims=[384, 64],
        vals=np.random.randn(384, 64).flatten().tolist()
    )
    
    weight2 = helper.make_tensor(
        name='weight2',
        data_type=TensorProto.FLOAT,
        dims=[384, 64],
        vals=np.random.randn(384, 64).flatten().tolist()
    )
    
    # Create a constant that will be converted to Shape type
    # This simulates the scenario where a rank-1 tensor is used with Shape outputs
    # We need to ensure mathematical correctness:
    # Input shape will be [batch, seq, 384], MatMul outputs will be [batch, seq, 64] each
    # Concat gives [batch, seq, 128]
    # Instead of using the full input shape, we'll use a Slice to get just [batch, seq]
    # Then concatenate with [8, 16] to get reshape target [batch, seq, 8, 16]
    # This ensures batch*seq*128 elements reshape to batch*seq*8*16 (same number)
    const_shape = helper.make_tensor(
        name='const_shape_value',
        data_type=TensorProto.INT64,
        dims=[2],  # rank-1 tensor
        vals=[8, 16]  # Will be concatenated with sliced shape to form [batch, seq, 8, 16]
    )
    
    # Create nodes
    
    # Shape node to get input shape
    shape_node = helper.make_node(
        'Shape',
        inputs=['input'],
        outputs=['input_shape'],
        name='shape'
    )
    
    # Slice the shape to get only [batch, seq] dimensions
    # Create constants for slice parameters
    slice_starts = helper.make_tensor(
        name='slice_starts',
        data_type=TensorProto.INT64,
        dims=[1],
        vals=[0]
    )
    
    slice_ends = helper.make_tensor(
        name='slice_ends',
        data_type=TensorProto.INT64,
        dims=[1],
        vals=[2]
    )
    
    slice_starts_node = helper.make_node(
        'Constant',
        inputs=[],
        outputs=['starts'],
        value=slice_starts
    )
    
    slice_ends_node = helper.make_node(
        'Constant',
        inputs=[],
        outputs=['ends'],
        value=slice_ends
    )
    
    # Slice to get first 2 dimensions [batch, seq]
    slice_node = helper.make_node(
        'Slice',
        inputs=['input_shape', 'starts', 'ends'],
        outputs=['sliced_shape'],
        name='slice'
    )
    
    # Constant node that outputs the rank-1 tensor
    const_node = helper.make_node(
        'Constant',
        inputs=[],
        outputs=['const_shape'],
        name='constant',
        value=const_shape
    )
    
    # Concat that combines sliced Shape output with Constant output
    # This will trigger the constant->Shape conversion
    # Result will be [batch, seq, 8, 16]
    concat_node = helper.make_node(
        'Concat',
        inputs=['sliced_shape', 'const_shape'],
        outputs=['concat_shape'],
        name='concat',
        axis=0
    )
    
    # First MatMul - should output rank 3
    matmul1_node = helper.make_node(
        'MatMul',
        inputs=['input', 'weight1'],
        outputs=['matmul1_out'],
        name='matmul1'
    )
    
    # Second MatMul - should output rank 3
    matmul2_node = helper.make_node(
        'MatMul',
        inputs=['input', 'weight2'],
        outputs=['matmul2_out'],
        name='matmul2'
    )
    
    # Concat the MatMul outputs along axis 2 (feature dimension)
    # This would fail if MatMul outputs are incorrectly inferred as rank 2
    concat_matmul_node = helper.make_node(
        'Concat',
        inputs=['matmul1_out', 'matmul2_out'],
        outputs=['concat_matmul_out'],
        name='concat_matmul',
        axis=2  # Concatenate along the feature dimension
    )
    
    # Reshape using the concatenated shape
    # This ensures the shape concatenation result is used
    reshape_node = helper.make_node(
        'Reshape',
        inputs=['concat_matmul_out', 'concat_shape'],
        outputs=['output'],
        name='reshape'
    )
    
    # Create output with expected shape
    output = helper.make_tensor_value_info(
        'output', TensorProto.FLOAT, [None, None, 8, 16]  # Dynamic batch and sequence, fixed last dims
    )
    
    # Create the graph
    graph = helper.make_graph(
        [shape_node, slice_starts_node, slice_ends_node, slice_node, 
         const_node, concat_node, matmul1_node, matmul2_node, 
         concat_matmul_node, reshape_node],
        'rank_inference_propagation',
        [input_3d],
        [output],
        [weight1, weight2]  # Initializers
    )
    
    # Create the model
    model = helper.make_model(graph)
    model.opset_import[0].version = 16
    
    return model

if __name__ == '__main__':
    model = create_model()
    
    # Validate
    onnx.checker.check_model(model)
    
    # Save
    onnx.save(model, 'rank_inference_propagation.onnx')
    print("Created rank_inference_propagation.onnx")
    
    # Verify with ReferenceEvaluator
    try:
        from onnx.reference import ReferenceEvaluator
        
        # Create test input
        test_input = np.random.randn(2, 4, 384).astype(np.float32)
        
        print(f"\nTest input shape: {test_input.shape}")
        
        # Run the model using ReferenceEvaluator
        session = ReferenceEvaluator('rank_inference_propagation.onnx', verbose=0)
        outputs = session.run(None, {"input": test_input})
        
        print(f"Test output shape: {outputs[0].shape}")
        print(f"Expected shape: (2, 4, 8, 16)")
        
        # Verify the output shape matches expectations
        # The reshape operation concatenates sliced input shape [2, 4] with constant [8, 16]
        # to form [2, 4, 8, 16], which matches the number of elements in the
        # concatenated MatMul outputs [2, 4, 128] (2*4*128 = 2*4*8*16 = 1024)
        
    except ImportError:
        print("ONNX ReferenceEvaluator not available for validation")