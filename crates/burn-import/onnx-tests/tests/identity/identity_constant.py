#!/usr/bin/env python3

import onnx
import numpy as np
from onnx import helper, TensorProto

def main():
    """Create an Identity node with a constant input - should be converted to Constant node"""
    
    # Create a constant tensor
    constant_data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    constant_tensor = helper.make_tensor(
        name='const_data',
        data_type=TensorProto.FLOAT,
        dims=constant_data.shape,
        vals=constant_data.flatten()
    )
    
    # Create Identity node with constant input
    identity_node = helper.make_node(
        'Identity',
        inputs=['const_data'],
        outputs=['identity_output'],
        name='identity_with_constant'
    )
    
    # Define the output
    output_tensor = helper.make_tensor_value_info('identity_output', TensorProto.FLOAT, [3])
    
    # Create the graph
    graph = helper.make_graph(
        nodes=[identity_node],
        name='IdentityConstantTest',
        inputs=[],  # No inputs - constant is from initializer
        outputs=[output_tensor],
        initializer=[constant_tensor]
    )
    
    # Create the model
    model = helper.make_model(
        graph, 
        producer_name='test',
        opset_imports=[helper.make_opsetid("", 16)]
    )
    
    # Save the model
    onnx.save(model, "identity_constant.onnx")
    print("Generated identity_constant.onnx - Identity node with constant input")

if __name__ == "__main__":
    main()