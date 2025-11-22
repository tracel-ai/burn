#!/usr/bin/env -S uv run
# /// script
# dependencies = [
#   "onnx>=1.15.0",
#   "numpy>=1.24.0",
# ]
# ///

"""Generate ONNX model with no initializers (all runtime inputs)."""

import onnx
from onnx import helper, TensorProto

def create_model():
    input1 = helper.make_tensor_value_info('input1', TensorProto.FLOAT, [1, 3])
    input2 = helper.make_tensor_value_info('input2', TensorProto.FLOAT, [1, 3])
    output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 3])
    
    nodes = [
        helper.make_node('Add', ['input1', 'input2'], ['temp'], name='add'),
        helper.make_node('Relu', ['temp'], ['output'], name='relu'),
    ]
    
    graph = helper.make_graph(nodes, 'no_initializers', [input1, input2], [output])
    model = helper.make_model(graph, producer_name="onnx-ir-test", opset_imports=[helper.make_opsetid("", 16)])
    onnx.checker.check_model(model)
    return model

if __name__ == '__main__':
    model = create_model()
    onnx.save(model, '../fixtures/no_initializers.onnx')
    print("Model saved to ../fixtures/no_initializers.onnx")
    print("  No initializers - all inputs are runtime")
