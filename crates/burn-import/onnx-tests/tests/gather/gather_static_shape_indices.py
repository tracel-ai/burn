#!/usr/bin/env python3

# used to generate model: gather_static_shape_indices.onnx
# Test case that demonstrates why shape preservation is needed for static indices
# This creates a model with static indices that have specific shapes

import onnx
from onnx import helper, TensorProto
from onnx.reference import ReferenceEvaluator
import numpy as np


def create_test_with_constant_indices():
    """Create an ONNX model with pre-defined constant indices to test shape inference"""
    
    # Create input
    data = helper.make_tensor_value_info('data', TensorProto.FLOAT, [3, 4, 5])
    
    # Create different shaped constant indices with same/similar values
    # Scalar index (rank 0) - single value 1
    scalar_indices = helper.make_tensor('scalar_indices', TensorProto.INT64, [], [1])
    
    # 1D index (rank 1) - array [1]  
    indices_1d = helper.make_tensor('indices_1d', TensorProto.INT64, [1], [1])
    
    # 2D index (rank 2) - array [[1, 0]]
    indices_2d = helper.make_tensor('indices_2d', TensorProto.INT64, [1, 2], [1, 0])
    
    # Create Gather nodes
    gather_scalar = helper.make_node(
        'Gather',
        inputs=['data', 'scalar_indices'],
        outputs=['output_scalar'],
        axis=0
    )
    
    gather_1d = helper.make_node(
        'Gather',
        inputs=['data', 'indices_1d'],
        outputs=['output_1d'],
        axis=0
    )
    
    gather_2d = helper.make_node(
        'Gather',
        inputs=['data', 'indices_2d'],
        outputs=['output_2d'],
        axis=0
    )
    
    # Create outputs with expected shapes
    # scalar: [3,4,5] with scalar index -> [4,5]
    output_scalar = helper.make_tensor_value_info('output_scalar', TensorProto.FLOAT, [4, 5])
    # 1d: [3,4,5] with [1] index -> [1,4,5]
    output_1d = helper.make_tensor_value_info('output_1d', TensorProto.FLOAT, [1, 4, 5])
    # 2d: [3,4,5] with [1,2] index -> [1,2,4,5]
    output_2d = helper.make_tensor_value_info('output_2d', TensorProto.FLOAT, [1, 2, 4, 5])
    
    # Create the graph
    graph = helper.make_graph(
        [gather_scalar, gather_1d, gather_2d],
        'gather_rank_test',
        [data],
        [output_scalar, output_1d, output_2d],
        [scalar_indices, indices_1d, indices_2d]
    )
    
    # Create the model
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 16)])
    
    # Save the model
    onnx.save(model, 'gather_static_shape_indices.onnx')
    print("Created gather_static_shape_indices.onnx")
    print("\nExpected output shapes:")
    print("- Scalar index (rank 0): [3,4,5] -> [4,5]")
    print("- 1D index (rank 1): [3,4,5] -> [1,4,5]") 
    print("- 2D index (rank 2): [3,4,5] -> [1,2,4,5]")
    print("\nWithout shape preservation, we can't distinguish between these cases!")
    
    # Verify the model using ReferenceEvaluator
    print("\n=== Verifying model with ReferenceEvaluator ===")
    sess = ReferenceEvaluator(model)
    
    # Create test input data
    test_data = np.ones((3, 4, 5), dtype=np.float32)
    # Make index 1 different so we can verify correct indexing
    test_data[1, :, :] = 5.0
    
    # Run the model
    outputs = sess.run(None, {'data': test_data})
    
    # Verify outputs
    output_scalar = outputs[0]
    output_1d = outputs[1]
    output_2d = outputs[2]
    
    print(f"Actual output shapes:")
    print(f"- Scalar index output shape: {output_scalar.shape}")
    print(f"- 1D index output shape: {output_1d.shape}")
    print(f"- 2D index output shape: {output_2d.shape}")
    
    # Verify shapes match expectations
    assert output_scalar.shape == (4, 5), f"Scalar output shape {output_scalar.shape} != (4, 5)"
    assert output_1d.shape == (1, 4, 5), f"1D output shape {output_1d.shape} != (1, 4, 5)"
    assert output_2d.shape == (1, 2, 4, 5), f"2D output shape {output_2d.shape} != (1, 2, 4, 5)"
    
    # Verify values - all should have selected index 1 (value 5.0)
    assert np.all(output_scalar == 5.0), "Scalar output should be all 5.0"
    assert np.all(output_1d == 5.0), "1D output should be all 5.0"
    assert np.all(output_2d[0, 0] == 5.0), "2D output first index should be all 5.0"
    assert np.all(output_2d[0, 1] == 1.0), "2D output second index should be all 1.0"
    
    print("✓ All outputs match expected shapes and values!")
    
    return model


if __name__ == "__main__":
    model = create_test_with_constant_indices()
    print("\n✓ Model successfully created and verified!")