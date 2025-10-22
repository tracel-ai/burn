#!/usr/bin/env -S uv run
# /// script
# dependencies = [
#   "onnx>=1.15.0",
#   "numpy>=1.24.0",
# ]
# ///

"""Generate ONNX model with mixed value sources (Static + Dynamic)."""

import onnx
from onnx import helper, TensorProto
import numpy as np

def create_model():
    input_runtime = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 3])
    output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 3])
    
    # Static constant
    bias = helper.make_tensor(
        name='bias',
        data_type=TensorProto.FLOAT,
        dims=[1, 3],
        vals=np.array([[1.0, 2.0, 3.0]], dtype=np.float32).flatten().tobytes(),
        raw=True
    )
    
    # Mix: runtime input (Dynamic) + constant (Static after lifting)
    nodes = [helper.make_node('Add', ['input', 'bias'], ['output'], name='add')]
    
    graph = helper.make_graph(nodes, 'mixed_value_sources', [input_runtime], [output], initializer=[bias])
    model = helper.make_model(graph, producer_name="onnx-ir-test", opset_imports=[helper.make_opsetid("", 16)])
    onnx.checker.check_model(model)
    return model

if __name__ == '__main__':
    model = create_model()
    onnx.save(model, '../fixtures/mixed_value_sources.onnx')
    print("Model saved to ../fixtures/mixed_value_sources.onnx")
    print("  Add with Dynamic input + Static constant")
