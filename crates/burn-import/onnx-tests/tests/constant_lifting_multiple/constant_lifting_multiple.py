#!/usr/bin/env python3

import onnx
import numpy as np
from onnx import helper, TensorProto

def main():
    """Create a Clip node with multiple constant inputs that should be lifted"""
    
    # Define the input tensor
    input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, [2, 3])
    
    # Create min constant (this should be lifted to a Constant node)
    min_constant = helper.make_node(
        'Constant',
        inputs=[],
        outputs=['min_value'],
        value=helper.make_tensor(
            name='min_tensor',
            data_type=TensorProto.FLOAT,
            dims=[],
            vals=[0.0]
        ),
        name='min_constant'
    )
    
    # Create max constant (this should be lifted to a Constant node)  
    max_constant = helper.make_node(
        'Constant',
        inputs=[],
        outputs=['max_value'],
        value=helper.make_tensor(
            name='max_tensor',
            data_type=TensorProto.FLOAT,
            dims=[],
            vals=[6.0]
        ),
        name='max_constant'
    )
    
    # Create Clip node that uses both constants
    # Clip is in LIFT_CONSTANTS_FOR_NODE_TYPES, so constants should be lifted
    clip_node = helper.make_node(
        'Clip',
        inputs=['input', 'min_value', 'max_value'],
        outputs=['clipped_output'],
        name='clip_with_constants'
    )
    
    # Define the output
    output_tensor = helper.make_tensor_value_info('clipped_output', TensorProto.FLOAT, [2, 3])
    
    # Create the graph
    graph = helper.make_graph(
        nodes=[min_constant, max_constant, clip_node],
        name='ConstantLiftingMultipleTest',
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
    onnx.save(model, "constant_lifting_multiple.onnx")
    print("Generated constant_lifting_multiple.onnx - Clip node with multiple constants to lift")

if __name__ == "__main__":
    main()