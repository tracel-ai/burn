#!/usr/bin/env python3

# used to generate model: onnx-tests/tests/range/range.onnx

import onnx
from onnx import helper, TensorProto

def main():
    node = onnx.helper.make_node(
        'Range',
        name='range',
        inputs=['start', 'end', 'step'],
        outputs=['output']
    )

    graph_def = helper.make_graph(
        nodes=[node],
        name='RangeGraph',
        inputs=[
            helper.make_tensor_value_info('start', TensorProto.INT64, []),
            helper.make_tensor_value_info('end', TensorProto.INT64, []),
            helper.make_tensor_value_info('step', TensorProto.INT64, [])
        ],
        outputs=[
            helper.make_tensor_value_info('output', TensorProto.INT64, [5])
        ],
    )

    model_def = helper.make_model(graph_def, producer_name='range')

    onnx.save(model_def, 'range.onnx')

if __name__ == '__main__':
    main()
