#!/usr/bin/env python3

import onnx
import numpy as np
from onnx import helper, TensorProto

def main():
    """Create multiple Identity nodes in sequence - should be optimized"""
    
    # Define input and output
    input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, [2, 3])
    output_tensor = helper.make_tensor_value_info('final_output', TensorProto.FLOAT, [2, 3])
    
    # Create chain of Identity nodes
    identity1_node = helper.make_node(
        'Identity',
        inputs=['input'],
        outputs=['identity1_out'],
        name='identity1'
    )
    
    identity2_node = helper.make_node(
        'Identity',
        inputs=['identity1_out'],
        outputs=['identity2_out'],
        name='identity2'
    )
    
    # Final operation to produce output
    relu_node = helper.make_node(
        'Relu',
        inputs=['identity2_out'],
        outputs=['final_output'],
        name='relu_final'
    )
    
    # Create the graph
    graph = helper.make_graph(
        nodes=[identity1_node, identity2_node, relu_node],
        name='IdentityChainTest',
        inputs=[input_tensor],
        outputs=[output_tensor]
    )
    
    # Create the model
    model = helper.make_model(
        graph, 
        producer_name='test',
        opset_imports=[helper.make_operatorsetid("", 16)]
    )
    
    # Save the model
    onnx.save(model, "identity_chain.onnx")
    print("Generated identity_chain.onnx - Multiple Identity nodes in sequence")

if __name__ == "__main__":
    main()