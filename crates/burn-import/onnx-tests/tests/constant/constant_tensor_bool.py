#!/usr/bin/env python3

import onnx
import onnx.helper as helper
import onnx.checker as checker
import numpy as np
from onnx import TensorProto

def create_model():
    """Create an ONNX model with a multidimensional bool tensor constant."""
    
    # Create a 2x3 bool constant tensor
    constant_value = np.array([[True, False, True], [False, True, False]], dtype=bool)
    
    # Create constant node
    constant_node = helper.make_node(
        'Constant',
        inputs=[],
        outputs=['constant_tensor'],
        value=helper.make_tensor(
            name='const_tensor_value',
            data_type=TensorProto.BOOL,
            dims=constant_value.shape,
            vals=constant_value.flatten().astype(int).tolist()
        )
    )
    
    # Create Or node to combine the constant with input (bool operations)
    or_node = helper.make_node(
        'Or',
        inputs=['input', 'constant_tensor'],
        outputs=['output']
    )
    
    # Create graph
    graph_def = helper.make_graph(
        [constant_node, or_node],
        'constant_tensor_bool_model',
        [helper.make_tensor_value_info('input', TensorProto.BOOL, [2, 3])],
        [helper.make_tensor_value_info('output', TensorProto.BOOL, [2, 3])]
    )
    
    # Create model with opset version 16
    model_def = helper.make_model(
        graph_def, 
        producer_name='constant_tensor_bool_test',
        opset_imports=[helper.make_opsetid("", 16)]
    )
    
    # Check model
    checker.check_model(model_def)
    
    return model_def

if __name__ == '__main__':
    model = create_model()
    onnx.save(model, 'constant_tensor_bool.onnx')
    print("Created constant_tensor_bool.onnx")