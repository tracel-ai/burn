#!/usr/bin/env -S uv run
# /// script
# dependencies = [
#   "onnx>=1.15.0",
#   "numpy>=1.24.0",
# ]
# ///

"""Generate ONNX model with high-rank tensors (5D, 6D)."""

import onnx
from onnx import helper, TensorProto

def create_model():
    input_5d = helper.make_tensor_value_info('input_5d', TensorProto.FLOAT, [2, 3, 4, 5, 6])
    input_6d = helper.make_tensor_value_info('input_6d', TensorProto.FLOAT, [1, 2, 3, 4, 5, 6])
    output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [2, 3, 4, 5, 6])
    
    nodes = [
        helper.make_node('Relu', ['input_5d'], ['relu_out'], name='relu_5d'),
        helper.make_node('Abs', ['input_6d'], ['abs_out'], name='abs_6d'),
        helper.make_node('Add', ['relu_out', 'abs_out'], ['output'], name='add'),
    ]
    
    graph = helper.make_graph(nodes, 'high_rank_model', [input_5d, input_6d], [output])
    model = helper.make_model(graph, producer_name="onnx-ir-test", opset_imports=[helper.make_opsetid("", 16)])
    onnx.checker.check_model(model)
    return model

if __name__ == '__main__':
    model = create_model()
    onnx.save(model, '../fixtures/high_rank_tensors.onnx')
    print("Model saved to ../fixtures/high_rank_tensors.onnx")
    print("  5D tensor: [2, 3, 4, 5, 6]")
    print("  6D tensor: [1, 2, 3, 4, 5, 6]")
