#!/usr/bin/env python3

# used to generate model: onnx-tests/tests/slice/slice_1d_tensor.onnx

import numpy as np
import onnx
from onnx import helper, TensorProto
from onnx.reference import ReferenceEvaluator

OPSET_VERSION = 18

def main():
    # Create input/output value infos
    input_tensor = helper.make_tensor_value_info(
        'input', TensorProto.FLOAT, [4, 5, 6]
    )
    start_tensor = helper.make_tensor_value_info(
        'starts', TensorProto.INT64, [2]  # 1D tensor with 2 elements
    )
    end_tensor = helper.make_tensor_value_info(
        'ends', TensorProto.INT64, [2]  # 1D tensor with 2 elements
    )
    output_tensor = helper.make_tensor_value_info(
        'output', TensorProto.FLOAT, [2, 3, 6]  # expected output shape
    )
    
    # Create axes constant (which dimensions to slice)
    axes_const = helper.make_tensor(
        name='axes',
        data_type=TensorProto.INT64,
        dims=[2],
        vals=[0, 1]  # slice along dimensions 0 and 1
    )
    
    # Create the slice node
    slice_node = helper.make_node(
        'Slice',
        inputs=['input', 'starts', 'ends', 'axes'],
        outputs=['output'],
        name='slice'
    )
    
    # Create the graph
    graph = helper.make_graph(
        nodes=[slice_node],
        name='slice_1d_tensor',
        inputs=[input_tensor, start_tensor, end_tensor],
        outputs=[output_tensor],
        initializer=[axes_const]
    )
    
    # Create the model
    model = helper.make_model(
        graph,
        producer_name='slice_1d_tensor_generator',
        opset_imports=[helper.make_operatorsetid("", OPSET_VERSION)]
    )
    
    # Check and save
    onnx.checker.check_model(model)
    onnx.save(model, 'slice_1d_tensor.onnx')
    
    # Create test data and run inference to print dynamic shapes
    session = ReferenceEvaluator(model)
    input_data = np.arange(1, 121).reshape(4, 5, 6).astype(np.float32)
    starts = np.array([1, 2], dtype=np.int64)
    ends = np.array([3, 5], dtype=np.int64)
    
    outputs = session.run(None, {
        'input': input_data,
        'starts': starts,
        'ends': ends
    })
    
    print("Finished exporting model to slice_1d_tensor.onnx")
    print("Model structure:")
    print(f"  Inputs: input{list(input_data.shape)}, starts{list(starts.shape)}, ends{list(ends.shape)}")
    print(f"  Outputs: output{list(outputs[0].shape)}")
    print(f"Test case: starts={starts.tolist()}, ends={ends.tolist()}")
    print(f"Result shape: {outputs[0].shape}")


if __name__ == '__main__':
    main()