#!/usr/bin/env python3

# Test case for Gather with different combinations of Initializer and Constant
# Creates a single model with 3 Gather nodes to test:
# 1. Both inputs as initializers
# 2. Data as initializer, indices as Constant node
# 3. Data as Constant node, indices as initializer

import onnx
from onnx import helper, TensorProto
from onnx.reference import ReferenceEvaluator
import numpy as np


def create_gather_constant_2d_indices():
    """Create an ONNX model with 3 Gather nodes using different input combinations"""
    
    # === Common data for all three cases ===
    # Create data values: float32[5,7]
    data_values = np.arange(35, dtype=np.float32).reshape(5, 7)
    
    # Create indices values: int64[1,5]
    indices_values = np.array([[0, 2, 1, 4, 3]], dtype=np.int64)
    
    # === Case 1: Both as initializers ===
    data_init_1 = helper.make_tensor(
        'data_init_1',
        TensorProto.FLOAT,
        [5, 7],
        data_values.flatten().tolist()
    )
    
    indices_init_1 = helper.make_tensor(
        'indices_init_1',
        TensorProto.INT64,
        [1, 5],
        indices_values.flatten().tolist()
    )
    
    gather_node_1 = helper.make_node(
        'Gather',
        inputs=['data_init_1', 'indices_init_1'],
        outputs=['output_1'],
        axis=0,
        name='Gather_both_init'
    )
    
    # === Case 2: Data as initializer, indices as Constant node ===
    data_init_2 = helper.make_tensor(
        'data_init_2',
        TensorProto.FLOAT,
        [5, 7],
        data_values.flatten().tolist()
    )
    
    # Create Constant node for indices
    indices_const_tensor = helper.make_tensor(
        'indices_const_tensor',
        TensorProto.INT64,
        [1, 5],
        indices_values.flatten().tolist()
    )
    
    const_node_indices = helper.make_node(
        'Constant',
        inputs=[],
        outputs=['indices_const'],
        value=indices_const_tensor,
        name='Constant_indices'
    )
    
    gather_node_2 = helper.make_node(
        'Gather',
        inputs=['data_init_2', 'indices_const'],
        outputs=['output_2'],
        axis=0,
        name='Gather_init_const'
    )
    
    # === Case 3: Data as Constant node, indices as initializer ===
    # Create Constant node for data
    data_const_tensor = helper.make_tensor(
        'data_const_tensor',
        TensorProto.FLOAT,
        [5, 7],
        data_values.flatten().tolist()
    )
    
    const_node_data = helper.make_node(
        'Constant',
        inputs=[],
        outputs=['data_const'],
        value=data_const_tensor,
        name='Constant_data'
    )
    
    indices_init_3 = helper.make_tensor(
        'indices_init_3',
        TensorProto.INT64,
        [1, 5],
        indices_values.flatten().tolist()
    )
    
    gather_node_3 = helper.make_node(
        'Gather',
        inputs=['data_const', 'indices_init_3'],
        outputs=['output_3'],
        axis=0,
        name='Gather_const_init'
    )
    
    # === Combine outputs with Add nodes to create single output ===
    # This ensures all three paths are executed
    # output = output_1 + output_2 + output_3
    add_node_1 = helper.make_node(
        'Add',
        inputs=['output_1', 'output_2'],
        outputs=['sum_12'],
        name='Add_1_2'
    )
    
    add_node_2 = helper.make_node(
        'Add',
        inputs=['sum_12', 'output_3'],
        outputs=['output'],
        name='Add_final'
    )
    
    # Create output tensor info
    output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 5, 7])
    
    # Create the graph with all nodes
    graph = helper.make_graph(
        [
            # Constant nodes
            const_node_indices,
            const_node_data,
            # Gather nodes
            gather_node_1,
            gather_node_2,
            gather_node_3,
            # Add nodes to combine outputs
            add_node_1,
            add_node_2
        ],
        'gather_constant_2d_indices',
        [],  # No external inputs
        [output],
        [
            # Initializers
            data_init_1,
            indices_init_1,
            data_init_2,
            indices_init_3
        ]
    )
    
    # Create the model
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 16)])
    
    # Save the model
    onnx.save(model, 'gather_constant_2d_indices.onnx')
    
    print("Created gather_constant_2d_indices.onnx with 3 Gather nodes")
    print("\nModel structure:")
    print("- Gather_both_init: Both data and indices as initializers")
    print("- Gather_init_const: Data as initializer, indices as Constant node")
    print("- Gather_const_init: Data as Constant node, indices as initializer")
    print("\nAll three Gather operations use:")
    print(f"- Data shape: [5, 7]")
    print(f"- Indices shape: [1, 5]")
    print(f"- Indices values: {indices_values}")
    print(f"- Each produces output shape: [1, 5, 7]")
    print("\nFinal output = sum of all three Gather outputs")
    print("This tests constant lifting and different input handling scenarios")
    
    # Expected output for each gather
    expected_single = np.take(data_values, indices_values[0], axis=0).reshape(1, 5, 7)
    # Final output is 3x the single output since we add them
    expected_final = expected_single * 3
    print(f"\nExpected final output shape: {expected_final.shape}")
    print(f"Expected first element (3 * 0.0): {expected_final[0, 0, 0]}")
    
    # Verify the model using ReferenceEvaluator
    print("\n=== Verifying model with ReferenceEvaluator ===")
    sess = ReferenceEvaluator(model)
    
    # Run the model (no inputs needed since everything is constant/initializer)
    outputs = sess.run(None, {})
    
    # Verify the output
    actual_output = outputs[0]
    print(f"Actual output shape: {actual_output.shape}")
    print(f"Actual first element: {actual_output[0, 0, 0]}")
    
    # Check if outputs match
    if np.allclose(actual_output, expected_final):
        print("✓ Output matches expected values!")
    else:
        print("✗ Output does not match expected values!")
        print(f"Max difference: {np.max(np.abs(actual_output - expected_final))}")
    
    return model


if __name__ == "__main__":
    model = create_gather_constant_2d_indices()
    print("\n✓ Model successfully created and verified!")