#!/usr/bin/env -S uv run
# /// script
# dependencies = [
#   "onnx>=1.15.0",
#   "numpy>=1.24.0",
# ]
# ///

"""Generate ONNX model where all initializers are used."""

import onnx
from onnx import helper, TensorProto
import numpy as np

def create_model():
    input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 3])
    output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 3])
    
    # All 3 constants will be used
    consts = []
    for i in range(3):
        const = helper.make_tensor(
            name=f'const{i}',
            data_type=TensorProto.FLOAT,
            dims=[1, 3],
            vals=np.array([[i + 1.0] * 3], dtype=np.float32).flatten().tobytes(),
            raw=True
        )
        consts.append(const)
    
    nodes = [
        helper.make_node('Add', ['input', 'const0'], ['temp1'], name='add1'),
        helper.make_node('Mul', ['temp1', 'const1'], ['temp2'], name='mul'),
        helper.make_node('Add', ['temp2', 'const2'], ['output'], name='add2'),
    ]
    
    graph = helper.make_graph(nodes, 'all_constants_used', [input_tensor], [output], initializer=consts)
    model = helper.make_model(graph, producer_name="onnx-ir-test", opset_imports=[helper.make_opsetid("", 16)])
    onnx.checker.check_model(model)
    return model

if __name__ == '__main__':
    model = create_model()
    onnx.save(model, '../fixtures/all_constants_used.onnx')
    print("Model saved to ../fixtures/all_constants_used.onnx")
    print("  All 3 constants are referenced - none should be removed")
