#!/usr/bin/env -S uv run
# /// script
# dependencies = [
#   "onnx>=1.15.0",
#   "numpy>=1.24.0",
# ]
# ///

"""Generate ONNX model with single operation node."""

import onnx
from onnx import helper, TensorProto

def create_model():
    input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 4])
    output_tensor = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 4])
    
    nodes = [helper.make_node('Relu', ['input'], ['output'], name='single_relu')]
    
    graph = helper.make_graph(nodes, 'single_node', [input_tensor], [output_tensor])
    model = helper.make_model(graph, producer_name="onnx-ir-test", opset_imports=[helper.make_opsetid("", 16)])
    onnx.checker.check_model(model)
    return model

if __name__ == '__main__':
    model = create_model()
    onnx.save(model, '../fixtures/single_node.onnx')
    print("Model saved to ../fixtures/single_node.onnx")
    print("  Minimal graph: single Relu operation")
