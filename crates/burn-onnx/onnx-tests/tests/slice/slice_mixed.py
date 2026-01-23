#!/usr/bin/env python3

# used to generate model: onnx-tests/tests/slice/slice_mixed.onnx

import onnx
from onnx import helper, TensorProto


def main():
    # Create input/output value infos
    input_tensor = helper.make_tensor_value_info(
        'input', TensorProto.FLOAT, [5, 3]
    )
    end_scalar = helper.make_tensor_value_info(
        'end', TensorProto.INT64, []  # scalar - runtime input
    )
    output_tensor = helper.make_tensor_value_info(
        'output', TensorProto.FLOAT, [None, 3]  # variable size for first dim
    )
    
    # Create static starts constant (start at index 1)
    starts_const = helper.make_tensor(
        name='starts',
        data_type=TensorProto.INT64,
        dims=[1],
        vals=[1]  # static start at index 1
    )
    
    # Create axes and steps constants
    axes_const = helper.make_tensor(
        name='axes',
        data_type=TensorProto.INT64,
        dims=[1],
        vals=[0]  # slice along dimension 0
    )
    
    steps_const = helper.make_tensor(
        name='steps', 
        data_type=TensorProto.INT64,
        dims=[1],
        vals=[1]  # step size 1
    )
    
    # Create the slice node directly
    slice_node = helper.make_node(
        'Slice',
        inputs=['input', 'starts', 'end', 'axes', 'steps'],
        outputs=['output'],
        name='slice'
    )
    
    # Create the graph
    graph = helper.make_graph(
        nodes=[slice_node],
        name='slice_mixed',
        inputs=[input_tensor, end_scalar],
        outputs=[output_tensor],
        initializer=[starts_const, axes_const, steps_const]
    )
    
    # Create the model
    model = helper.make_model(graph, producer_name='slice_mixed_generator')
    model.opset_import[0].version = 18
    
    # Check and save
    onnx.checker.check_model(model)
    onnx.save(model, 'slice_mixed.onnx')
    
    print("Finished exporting model to slice_mixed.onnx")
    print("Model structure:")
    print(f"  Inputs: input[5,3], end[] (runtime)")
    print(f"  Outputs: output[var,3]")
    print(f"  Nodes: {len(model.graph.node)} (just Slice)")
    print("Test case: input[1:end, :] -> output[end-1, 3] (static start, runtime end)")


if __name__ == '__main__':
    main()