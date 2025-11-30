#!/usr/bin/env -S uv run
# /// script
# dependencies = [
#   "onnx>=1.15.0",
#   "numpy>=1.24.0",
# ]
# ///

"""Generate ONNX model with zero-sized dimensions."""

import onnx
from onnx import helper, TensorProto

def create_model():
    input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, [2, 0, 3])
    output_tensor = helper.make_tensor_value_info('output', TensorProto.FLOAT, [2, 0, 3])
    
    nodes = [helper.make_node('Relu', ['input'], ['output'], name='relu')]
    
    graph = helper.make_graph(nodes, 'zero_sized_dims_model', [input_tensor], [output_tensor])
    model = helper.make_model(graph, producer_name="onnx-ir-test", opset_imports=[helper.make_opsetid("", 16)])
    onnx.checker.check_model(model)
    return model

if __name__ == '__main__':
    model = create_model()
    onnx.save(model, '../fixtures/zero_sized_dims.onnx')
    print("Model saved to ../fixtures/zero_sized_dims.onnx")
    print("  Shape: [2, 0, 3] - zero-sized middle dimension")
