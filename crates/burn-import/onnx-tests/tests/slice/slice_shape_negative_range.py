#!/usr/bin/env python3
import numpy as np
import onnx
from onnx import TensorProto, ValueInfoProto
from onnx.helper import make_model, make_node, make_graph, make_tensor_value_info

def main():
    # Create a Shape node
    shape_node = make_node(
        'Shape',
        inputs=['input'],
        outputs=['shape_out']
    )
    
    # Create Slice node with negative indices: slice[-3:-1] to get elements from 3rd last to 2nd last
    slice_node = make_node(
        'Slice',
        inputs=['shape_out', 'starts', 'ends', 'axes', 'steps'],
        outputs=['output']
    )
    
    # Create initializers for slice parameters
    starts = onnx.numpy_helper.from_array(np.array([-3], dtype=np.int64), name='starts')
    ends = onnx.numpy_helper.from_array(np.array([-1], dtype=np.int64), name='ends')
    axes = onnx.numpy_helper.from_array(np.array([0], dtype=np.int64), name='axes')
    steps = onnx.numpy_helper.from_array(np.array([1], dtype=np.int64), name='steps')
    
    # Create value infos for input and output
    input_info = make_tensor_value_info('input', TensorProto.FLOAT, [2, 3, 4, 5])
    output_info = make_tensor_value_info('output', TensorProto.INT64, [2])  # Shape(2) output
    
    # Create the graph
    graph = make_graph(
        [shape_node, slice_node],
        'SliceShapeNegativeRange',
        [input_info],
        [output_info],
        [starts, ends, axes, steps]
    )
    
    # Create the model
    model = make_model(graph)
    model.opset_import[0].version = 16
    
    # Save the model
    onnx.save(model, 'slice_shape_negative_range.onnx')
    print(f"Model saved to slice_shape_negative_range.onnx")

if __name__ == '__main__':
    main()