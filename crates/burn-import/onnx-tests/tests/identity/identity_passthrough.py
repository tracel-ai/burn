#!/usr/bin/env python3

import onnx
import numpy as np
from onnx import helper, TensorProto

def main():
    """Create an Identity node acting as passthrough - should be removed"""
    
    # Define input and output
    input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, [2, 3])
    output_tensor = helper.make_tensor_value_info('final_output', TensorProto.FLOAT, [2, 3])
    
    # Create Identity node that passes through input
    identity_node = helper.make_node(
        'Identity',
        inputs=['input'],
        outputs=['identity_out'],
        name='identity_passthrough'
    )
    
    # Add another node to consume the identity output
    add_constant = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], dtype=np.float32)
    add_tensor = helper.make_tensor(
        name='add_const',
        data_type=TensorProto.FLOAT,
        dims=add_constant.shape,
        vals=add_constant.flatten()
    )
    
    add_node = helper.make_node(
        'Add',
        inputs=['identity_out', 'add_const'],
        outputs=['final_output'],
        name='add_after_identity'
    )
    
    # Create the graph
    graph = helper.make_graph(
        nodes=[identity_node, add_node],
        name='IdentityPassthroughTest',
        inputs=[input_tensor],
        outputs=[output_tensor],
        initializer=[add_tensor]
    )
    
    # Create the model
    model = helper.make_model(
        graph, 
        producer_name='test',
        opset_imports=[helper.make_opsetid("", 16)]
    )
    
    # Save the model
    onnx.save(model, "identity_passthrough.onnx")
    print("Generated identity_passthrough.onnx - Identity node as passthrough")

if __name__ == "__main__":
    main()