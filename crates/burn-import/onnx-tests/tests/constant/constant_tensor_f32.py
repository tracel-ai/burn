#!/usr/bin/env python3

import onnx
import onnx.helper as helper
import onnx.checker as checker
import numpy as np
from onnx import TensorProto

def create_model():
    """Create an ONNX model with a multidimensional f32 tensor constant."""
    
    # Create a 2x3 float32 constant tensor
    constant_value = np.array([[1.5, 2.5, 3.5], [4.5, 5.5, 6.5]], dtype=np.float32)
    
    # Create constant node
    constant_node = helper.make_node(
        'Constant',
        inputs=[],
        outputs=['constant_tensor'],
        value=helper.make_tensor(
            name='const_tensor_value',
            data_type=TensorProto.FLOAT,
            dims=constant_value.shape,
            vals=constant_value.flatten().tolist()
        )
    )
    
    # Create Add node to add the constant to input
    add_node = helper.make_node(
        'Add',
        inputs=['input', 'constant_tensor'],
        outputs=['output']
    )
    
    # Create graph
    graph_def = helper.make_graph(
        [constant_node, add_node],
        'constant_tensor_f32_model',
        [helper.make_tensor_value_info('input', TensorProto.FLOAT, [2, 3])],
        [helper.make_tensor_value_info('output', TensorProto.FLOAT, [2, 3])]
    )
    
    # Create model with opset version 16
    model_def = helper.make_model(
        graph_def, 
        producer_name='constant_tensor_f32_test',
        opset_imports=[helper.make_opsetid("", 16)]
    )
    
    # Check model
    checker.check_model(model_def)
    
    return model_def

if __name__ == '__main__':
    model = create_model()
    onnx.save(model, 'constant_tensor_f32.onnx')
    print("Created constant_tensor_f32.onnx")