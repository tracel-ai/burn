#!/usr/bin/env python3

# Tests constant lifting with various usage patterns including:
# - Constants used multiple times (should NOT be lifted)
# - Constants used once (should be lifted)
# - Mixed scenarios

import onnx
import numpy as np
from onnx import helper, TensorProto
from onnx.reference import ReferenceEvaluator

def main():
    """Create a model that tests constant lifting with various usage patterns"""
    
    # Define the input tensor
    input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, [2, 3])
    
    # Create a constant that will be used ONCE (should be lifted)
    single_use_constant = helper.make_node(
        'Constant',
        inputs=[],
        outputs=['single_use_value'],
        value=helper.make_tensor(
            name='single_use_tensor',
            data_type=TensorProto.FLOAT,
            dims=[],
            vals=[2.0]
        ),
        name='single_use_constant'
    )
    
    # Create a constant that will be used MULTIPLE times (should NOT be lifted)
    multi_use_constant = helper.make_node(
        'Constant',
        inputs=[],
        outputs=['multi_use_value'],
        value=helper.make_tensor(
            name='multi_use_tensor',
            data_type=TensorProto.FLOAT,
            dims=[],
            vals=[3.0]
        ),
        name='multi_use_constant'
    )
    
    # Create another single-use constant for clip operation
    clip_min_constant = helper.make_node(
        'Constant',
        inputs=[],
        outputs=['clip_min'],
        value=helper.make_tensor(
            name='clip_min_tensor',
            data_type=TensorProto.FLOAT,
            dims=[],
            vals=[0.0]
        ),
        name='clip_min_constant'
    )
    
    clip_max_constant = helper.make_node(
        'Constant',
        inputs=[],
        outputs=['clip_max'],
        value=helper.make_tensor(
            name='clip_max_tensor',
            data_type=TensorProto.FLOAT,
            dims=[],
            vals=[10.0]
        ),
        name='clip_max_constant'
    )
    
    # First operation: Mul with single-use constant
    # single_use_value is used only here, so it should be lifted
    mul1_node = helper.make_node(
        'Mul',
        inputs=['input', 'single_use_value'],
        outputs=['mul1_output'],
        name='mul_single_use'
    )
    
    # Second operation: Mul with multi-use constant (first use)
    # multi_use_value is used here and below, so it should NOT be lifted
    mul2_node = helper.make_node(
        'Mul',
        inputs=['mul1_output', 'multi_use_value'],
        outputs=['mul2_output'],
        name='mul_multi_use_1'
    )
    
    # Third operation: Add with the SAME multi-use constant (second use)
    # This ensures multi_use_value is referenced twice
    add_node = helper.make_node(
        'Add',
        inputs=['mul2_output', 'multi_use_value'],
        outputs=['add_output'],
        name='add_multi_use_2'
    )
    
    # Fourth operation: Clip with single-use constants
    # Both clip_min and clip_max are used only once, so they should be lifted
    clip_node = helper.make_node(
        'Clip',
        inputs=['add_output', 'clip_min', 'clip_max'],
        outputs=['final_output'],
        name='clip_with_constants'
    )
    
    # Define the output
    output_tensor = helper.make_tensor_value_info('final_output', TensorProto.FLOAT, [2, 3])
    
    # Create the graph
    graph = helper.make_graph(
        nodes=[
            single_use_constant,
            multi_use_constant,
            clip_min_constant,
            clip_max_constant,
            mul1_node,
            mul2_node,
            add_node,
            clip_node
        ],
        name='ConstantReusedTest',
        inputs=[input_tensor],
        outputs=[output_tensor]
    )
    
    # Create the model
    model = helper.make_model(
        graph, 
        producer_name='test',
        opset_imports=[helper.make_opsetid("", 16)]
    )
    
    # Save the model
    onnx.save(model, "constant_reused.onnx")
    print("Generated constant_reused.onnx - Tests constant lifting with reused constants")
    
    # Test the model with ReferenceEvaluator
    print("\nTesting with ReferenceEvaluator:")
    test_input = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    
    sess = ReferenceEvaluator(model)
    outputs = sess.run(None, {"input": test_input})
    
    print(f"Test input shape: {test_input.shape}")
    print(f"Test input: {test_input}")
    print(f"Expected computation:")
    print(f"  1. mul1 = input * 2.0 (single-use constant, should be lifted)")
    print(f"  2. mul2 = mul1 * 3.0 (multi-use constant, should NOT be lifted)")  
    print(f"  3. add = mul2 + 3.0 (same multi-use constant)")
    print(f"  4. output = clip(add, 0.0, 10.0) (single-use constants, should be lifted)")
    print(f"Final output: {outputs[0]}")
    
    # Manual verification
    mul1 = test_input * 2.0
    mul2 = mul1 * 3.0
    add = mul2 + 3.0
    expected = np.clip(add, 0.0, 10.0)
    print(f"Manually computed: {expected}")
    print(f"Match: {np.allclose(outputs[0], expected)}")

if __name__ == "__main__":
    main()