#!/usr/bin/env -S uv run
# /// script
# dependencies = [
#   "onnx>=1.15.0",
#   "numpy>=1.24.0",
# ]
# ///

"""Generate ONNX model with Gemm that should convert to Linear."""

import onnx
from onnx import helper, TensorProto
import numpy as np

def create_model():
    input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, [2, 3])
    output_tensor = helper.make_tensor_value_info('output', TensorProto.FLOAT, [2, 4])
    
    # Weight matrix for Gemm
    weight = helper.make_tensor(
        name='weight',
        data_type=TensorProto.FLOAT,
        dims=[3, 4],
        vals=np.random.randn(3, 4).astype(np.float32).tobytes(),
        raw=True
    )
    
    # Bias
    bias = helper.make_tensor(
        name='bias',
        data_type=TensorProto.FLOAT,
        dims=[4],
        vals=np.random.randn(4).astype(np.float32).tobytes(),
        raw=True
    )
    
    # Gemm: Y = alpha * A * B + beta * C
    # With alpha=1.0, beta=1.0, transA=0, transB=0 → can convert to Linear
    nodes = [
        helper.make_node(
            'Gemm',
            ['input', 'weight', 'bias'],
            ['output'],
            name='gemm',
            alpha=1.0,
            beta=1.0,
            transA=0,
            transB=0
        )
    ]
    
    graph = helper.make_graph(nodes, 'gemm_linear', [input_tensor], [output_tensor], initializer=[weight, bias])
    model = helper.make_model(graph, producer_name="onnx-ir-test", opset_imports=[helper.make_opsetid("", 16)])
    onnx.checker.check_model(model)
    return model

if __name__ == '__main__':
    model = create_model()
    onnx.save(model, '../fixtures/gemm_linear.onnx')
    print("Model saved to ../fixtures/gemm_linear.onnx")
    print("  Gemm that can be converted to Linear in Phase 2")
